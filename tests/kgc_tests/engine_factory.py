from dataset import KGCDataset
from keras_ns.grounding.backward_chaining_grounder import BackwardChainingGrounder
from keras_ns.grounding.domain_grounder import DomainGrounder
from keras_ns.grounding.engine import Engine
from keras_ns.grounding.file_grounder import FileGrounder
from keras_ns.grounding.relation_entity_grounder import RelationEntityGraphGrounder
from keras_ns.grounding.substitution_grounder import LocalFlatGrounder
from keras_ns.logic.commons import Rule
from typing import List, Tuple, Dict

class EngineFactory(object):

    # Static factory.
    @staticmethod
    def build(name: str,
              rules: List[Rule]=None,
              facts: List[Tuple]=None,
              domains: Dict[str, List[str]]=None,
              num_steps: int=1,
              n_threads: int=1,
              dataset: KGCDataset=None,
              max_elements: int=None,
              filepath: str=None) -> Engine:
        if name == "backward_chaining":
            assert dataset
            assert rules
            assert facts
            return BackwardChainingGrounder(rules, facts,
                                            num_steps,
                                            n_threads=n_threads)
        elif name == "full":
            assert dataset
            assert rules
            # A single default domain is assigned to all constants.
            if domains is None:
                domains = { Rule.default_domain(): dataset.constants }
            return DomainGrounder(domains=domains, rules=rules)
        elif name == "file":
            assert filepath
            assert rules
            return FileGrounder(filepath, rules)
        elif name == "local":
            assert rules
            return LocalFlatGrounder(rules)
        elif name == "full_on_relation_entity_graph":
            assert rules
            assert facts
            return RelationEntityGraphGrounder(rules, facts,
                                               max_elements=max_elements)
        else:
            raise Exception(False, ')Unknown Engine % s' % name)
