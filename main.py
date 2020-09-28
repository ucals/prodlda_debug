import math
import pyro
import pyro.distributions as dist
import torch
from torch.distributions import constraints
import torch.nn as nn
import torch.nn.functional as F
from pyro.infer import SVI, Trace_ELBO
from tqdm import trange


class Encoder(nn.Module):
    def __init__(self, vocab_size, num_topics, hidden, dropout):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.fc1 = nn.Linear(vocab_size, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, num_topics)

    def forward(self, inputs):
        h = F.softplus(self.fc1(inputs))
        h = F.softplus(self.fc2(h))
        h = self.drop(h)
        return F.softmax(self.fc3(h), dim=1)


class ProdLDA(nn.Module):
    def __init__(self, vocab_size, num_topics, hidden, dropout, device):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_topics = num_topics
        self.inference_net = Encoder(vocab_size, num_topics, hidden, dropout)
        self.device = device

    def model(self, docs=None, doc_sum=None):
        # Globals.
        with pyro.plate("topics", self.num_topics):
            a = torch.tensor(1. / self.num_topics, device=self.device)
            b = torch.tensor(1., device=self.device)
            topic_weights = pyro.sample("topic_weights", dist.Gamma(a, b))

            alpha = torch.ones(self.vocab_size, device=self.device) / self.vocab_size
            topic_words = pyro.sample("topic_words", dist.Dirichlet(alpha))

        # Locals.
        # We will use nested plates. Pyro convention is to count from the right
        # by using negative indices like -1, -2. This means documents must be at
        # the rightmost dimension, followed by words. For this reason, we transpose
        # the data:
        docs = docs.transpose(0, 1)

        with pyro.plate('documents', docs.shape[-1]):
            doc_topics = pyro.sample("doc_topics", dist.Dirichlet(topic_weights))
            with pyro.plate("words", docs.shape[-2]):
                word_topics = pyro.sample("word_topics", dist.Categorical(doc_topics),
                                          infer={"enumerate": "parallel"})
                data = pyro.sample("doc_words", dist.Categorical(topic_words[word_topics]),
                                   obs=docs)

        return topic_words

    def guide(self, docs=None, doc_sum=None):
        # Use a conjugate guide for global variables.
        topic_weights_posterior = pyro.param(
            "topic_weights_posterior",
            lambda: torch.ones(self.num_topics, device=self.device),
            constraint=constraints.positive)

        topic_words_posterior = pyro.param(
            "topic_words_posterior",
            lambda: torch.ones(self.num_topics, self.vocab_size, device=self.device),
            constraint=constraints.greater_than(0.5))

        with pyro.plate("topics", self.num_topics):
            pyro.sample("topic_weights", dist.Gamma(topic_weights_posterior, 1.))
            pyro.sample("topic_words", dist.Dirichlet(topic_words_posterior))

        # Use an amortized guide for local variables.
        pyro.module("inference_net", self.inference_net)
        with pyro.plate("documents", doc_sum.shape[0]):
            doc_topics = self.inference_net(doc_sum)
            pyro.sample("doc_topics", dist.Delta(doc_topics, event_dim=1))


def train(device, docs, doc_sum, batch_size, learning_rate, num_epochs):
    # clear param store
    pyro.clear_param_store()

    prodLDA = ProdLDA(
        vocab_size=doc_sum.shape[1],
        num_topics=20,
        hidden=100,
        dropout=0.2,
        device=device
    )
    prodLDA.to(device)

    optimizer = pyro.optim.ClippedAdam({"lr": learning_rate})
    svi = SVI(prodLDA.model, prodLDA.guide, optimizer, loss=Trace_ELBO())
    num_batches = int(math.ceil(len(docs) / batch_size))

    for epoch in range(num_epochs):
        running_loss = 0.0
        num_preds = 0

        # Iterate over data.
        bar = trange(num_batches, desc=('Epoch %d' % epoch).ljust(10))
        for i in bar:
            batch_docs = docs[i * batch_size:(i + 1) * batch_size, :]
            batch_doc_sum = doc_sum[i * batch_size:(i + 1) * batch_size, :]
            loss = svi.step(batch_docs, batch_doc_sum)

            # statistics
            running_loss += loss / batch_doc_sum.size(0)
            num_preds += 1
            bar.set_postfix(loss='{:.2f}'.format(running_loss / num_preds))

        epoch_loss = running_loss / len(docs)
        bar.set_postfix(epoch_loss='{:.2f}'.format(epoch_loss))

    return prodLDA


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    docs = torch.load('docs.pt').float().to(device)
    doc_sum = torch.load('doc_sum.pt').float().to(device)

    trained_model = train(device, docs, doc_sum, 32, 1e-3, 20)
