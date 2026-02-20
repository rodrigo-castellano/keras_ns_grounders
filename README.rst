
Keras NS Grounders
==================

**Keras NS Grounders** is a Neural-Symbolic (NS) learning framework built on top of Keras/TensorFlow.
It combines **symbolic logic reasoning** (first-order logic rules, backward/forward chaining grounding)
with **neural learning** (knowledge graph embeddings, differentiable reasoning layers) in a unified
end-to-end differentiable architecture.


Architecture
------------

The framework is organized into three main layers:

- **Logic layer** (``ns_lib/logic/``): atoms, rules, domains, first-order logic formalism, t-norms (Gödel, Product)
- **Grounding layer** (``ns_lib/grounding/``): multiple grounding engines that convert symbolic rules into neural queries
- **Neural layer** (``ns_lib/nn/``): KGE embeddings (TransE, ComplEx, DistMult, RotatE) and reasoning layers (R2N, DCR)



Models
~~~~~~

- ``r2n``         — Relational-to-Neural reasoning layer
- ``dcr``         — Differentiable Concept Reasoning
- ``sbr``         — Semantic-Based Regularization


Installation
------------

.. code:: bash

    # Clone the repository
    git clone https://github.com/rodrigo-castellano/keras_ns_grounders.git
    cd keras_ns_grounders

    # Install in editable mode
    pip install -e ./


Quick Start
-----------

.. code:: bash

    # Run default experiment (countries_s1, DCR model, ComplEx KGE)
    cd experiments
    python runner.py

    # Override config via command line
    python runner.py --dataset_name kinship_family --model_name r2n --kge rotate --epochs 100


Experiments
-----------

Training
~~~~~~~~

All experiments are launched via ``experiments/runner.py``, which reads ``experiments/config.yaml``
and supports grid search over hyperparameters.

.. code:: bash

    cd experiments
    python runner.py --dataset_name countries_s1 --model_name dcr --kge complex

See ``run.sh`` for ready-to-use example commands across all supported datasets.

Configuration
~~~~~~~~~~~~~

Edit ``experiments/config.yaml`` to change default parameters:

.. code:: yaml

    dataset_name: countries_s1
    grounder:     backward_0_1
    model_name:   dcr
    kge:          complex
    epochs:       100
    batch_size:   256
    learning_rate: 0.01

Results
~~~~~~~

Metrics (Hits@1, Hits@3, Hits@10, MRR, AUC-PR) and logic formulas are saved in:

.. code:: text

    experiments/runs/<dataset>/<model>/<run_signature>/


Data
~~~~

Datasets are located under ``experiments/data/``:

+----------------------+----------------------------------------------+
| Dataset              | Description                                  |
+======================+==============================================+
| ``countries_s1/2/3`` | Geographic relations (easy → hard splits)    |
+----------------------+----------------------------------------------+
| ``nations``          | Country-level political/cultural relations   |
+----------------------+----------------------------------------------+
| ``kinship``          | Family/kinship relations                     |
+----------------------+----------------------------------------------+
| ``kinship_family``   | Extended kinship with family predicates      |
+----------------------+----------------------------------------------+
| ``wn18rr``           | WordNet 18-RR knowledge graph                |
+----------------------+----------------------------------------------+
| ``FB15k237``         | Freebase 15k-237 knowledge graph             |
+----------------------+----------------------------------------------+
| ``umls``             | Unified Medical Language System              |
+----------------------+----------------------------------------------+
| ``pharmkg_small``    | Pharmaceutical knowledge graph (small)       |
+----------------------+----------------------------------------------+

Each dataset folder contains:

- ``train.txt`` / ``valid.txt`` / ``test.txt`` — fact splits
- ``facts.txt``                                — full background knowledge
- ``rules.txt``                                — first-order logic rules
- ``domain2constants.txt``                     — domain → constants mapping

Data format (functional notation):

.. code:: text

    locatedInCR(afghanistan,asia).
    neighborOf(france,germany).

Rules format:

.. code:: text

    r0:1:locatedInCS(X,W), locatedInSR(W,Z) -> locatedInCR(X,Z)
    r1:1:neighborOf(X,Y), locatedInCR(Y,Z) -> locatedInCR(X,Z)


Authors
-------

Rodrigo Castellano Ontiveros, Michelangelo Diligenti et al.


Licence
-------

Copyright 2024

Licensed under the Apache License, Version 2.0 (the "License"); you may
not use this file except in compliance with the License. You may obtain
a copy of the License at: http://www.apache.org/licenses/LICENSE-2.0.

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

See the License for the specific language governing permissions and
limitations under the License.
