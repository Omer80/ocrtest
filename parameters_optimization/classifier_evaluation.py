from collections import namedtuple

ClassifierEvaluation = namedtuple('ClassifierEvaluation',
                                  ['name',
                                   'parameters',
                                   'optimization_method',
                                   'accuracy',
                                   'f1', 'precision', 'recall',
                                   'precision_positive', 'precision_negative',
                                   'recall_positive', 'recall_negative',
                                   'f1_positive', 'f1_negative',
                                   'support_positive', 'support_negative'
                                   ])
