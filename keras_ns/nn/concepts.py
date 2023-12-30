import keras
import tensorflow as tf
from collections import Counter
from keras.layers import Layer, Dense, Dropout, LeakyReLU
import numpy as np
from keras_ns.logic.semantics import GodelTNorm, Logic
from keras_ns.logic.commons import Rule


def softselect(values, temperature, axis=1):
    if temperature == 0.0:
        return tf.nn.softmax(values, axis=axis)
    else:
        softmax_scores = tf.nn.log_softmax(values, axis=axis)
        mean_softmax_score = tf.reduce_mean(softmax_scores, axis=axis, keepdims=True)
        # Since mean_softmax_score is a negative number,
        # -temperature * mean_softmax_score is a positive bias term.
        softscores = tf.nn.sigmoid(softmax_scores - temperature * mean_softmax_score)
        return softscores


class ConceptReasoningLayer(Layer):
    def __init__(self, rule: Rule, emb_size: int, n_classes: int,
                 logic: Logic = GodelTNorm(),
                 filter_num_heads: int=1,
                 filter_activity_regularization: float=0.0,
                 temperature: float=1.0,
                 signed=True):
        super().__init__()
        self.rule = rule
        self.emb_size = emb_size
        self.n_classes = n_classes
        self.logic = logic
        self.filter_num_heads = filter_num_heads
        filter_activity_regularizer = (
            tf.keras.regularizers.l2(filter_activity_regularization)
            if filter_activity_regularization > 0.0 else None)
        if self.filter_num_heads > 1:
            self.filter_nn = []
            for h in range(self.filter_num_heads):
                self.filter_nn.append(keras.Sequential([
                    Dense(emb_size, activation=None, name=('filter_%d_1' % h)),
                    LeakyReLU(),
                    Dense(n_classes, activation=None, name=('filter_%d_2' % h),
                          activity_regularizer=filter_activity_regularizer)]))
        else:
            self.filter_nn = keras.Sequential([
                    Dense(emb_size, activation=None, name='filter_1'),
                    LeakyReLU(),
                    Dense(n_classes, activation=None, name='filter_2',
                          activity_regularizer=filter_activity_regularizer)])

        self.signed = signed
        if self.signed:  # Keep only positive concepts.
            self.sign_nn = keras.Sequential([
                Dense(emb_size, activation=None, name='sign_1'),
                LeakyReLU(),
                Dense(n_classes, activation=None, name='sign_2')])
        else:
            self.sign_nn = None
        self.temperature = tf.Variable([temperature], dtype=tf.float32)
        self.max_aggregation = True

    def call(self, inputs, return_attn=False, *args, **kwargs):
        # x.shape = (batch_size, num_body_atoms, embedding_size)
        c, x = inputs
        values = tf.tile(c, multiples=(1, 1, self.n_classes))

        # compute attention scores to build logic sentence, each attention score will
        # represent whether the concept should be active or not in the logic sentence.
        if self.signed:
            sign_attn = tf.nn.sigmoid(self.sign_nn(x))
        else:  # Keep only positive concepts.
            sign_attn = tf.ones_like(values, dtype=tf.float32)
        sign_terms = self.logic.iff_pair(sign_attn, values)

        # Attention scores to identify only relevant concepts for a class
        if self.filter_num_heads > 1:
            filter_attns = []
            for h in range(self.filter_num_heads):
                filter_attn_h = self.filter_nn[h](x)
                # Softselect over the body atoms (axis=1).
                filter_attn_h = softselect(filter_attn_h, temperature=self.temperature, axis=1)
                filter_attns.append(filter_attn_h)
            filter_attn = tf.reduce_max(tf.stack(filter_attns, axis=0),
                                        axis=0, keepdims=False)
        else:
            filter_attn = self.filter_nn(x)
            # Softselect over the body atoms (axis=1).
            filter_attn = softselect(filter_attn, temperature=self.temperature, axis=1)
        # tf.print('FILTER_ATTN', tf.squeeze(filter_attn)[:10, ...])

        reg = tf.keras.regularizers.L2(0.001)
        self.add_loss(reg(self.temperature))

        # filtered implemented as "or(not a, b)", corresponding to "a -> b"
        filtered_values = self.logic.disj_pair(self.logic.neg(filter_attn),
                                               sign_terms)
        # filtered_values = self.logic.imply_pair(filter_attn, sign_terms)
        # tf.print('FILTER_VALUES', filtered_values, summarize=3)

        # generate minterm
        preds = tf.squeeze(self.logic.conj(filtered_values, axis=1), axis=1)
        # tf.print('PREDS', preds.shape, preds, summarize=3)

        if return_attn:
            return x, c, preds, sign_attn, filter_attn
        else:
            return preds

    def explain(self, x_unused, c, y_preds, sign_attn_mask, filter_attn_mask,
                indices_bodies, indices_head):
        mode = 'global'
        tf.print('TEMPERATURE', self.temperature)
        #tf.print('FILTER_ATTN', tf.squeeze(filter_attn_mask)[:10, ...])
        c = tf.cast(tf.squeeze(c), tf.float32)
        if self.max_aggregation:
            print(len(indices_head), '----------------------------------')
            idx_max_grounding = {}
            for idx_grounding, h in enumerate(indices_head):
                id_head = h[0].numpy()
                if id_head not in idx_max_grounding:
                    idx_max_grounding[id_head] = idx_grounding
                elif y_preds[idx_max_grounding[id_head]][0] < y_preds[idx_grounding][0]:
                    idx_max_grounding[id_head] = idx_grounding

            idx_max_grounding = tf.expand_dims(
                tf.constant(list(idx_max_grounding.values()), dtype=tf.int32),
                axis=-1)
            c = tf.gather_nd(c, idx_max_grounding)
            y_preds = tf.gather_nd(y_preds, idx_max_grounding)
            sign_attn_mask = tf.gather_nd(sign_attn_mask, idx_max_grounding)
            filter_attn_mask = tf.gather_nd(filter_attn_mask, idx_max_grounding)
            #tf.print('AGGR', 'FILT', filter_attn_mask, 'SIGN', sign_attn_mask,
            #         'PREDS', y_preds)

        concept_names = [str(a) for a in self.rule.body]
        class_names = [str(a) for a in self.rule.head]

        explanations = []
        all_class_explanations = {cn: [] for cn in class_names}
        for sample_idx in range(len(y_preds)):
            prediction = y_preds[sample_idx] > 0.5
            # Select 1 and flatten.
            active_classes = np.argwhere(prediction).reshape(-1)
            if len(active_classes) == 0:
                # print('Skipping ', sample_idx, 'PREDS', y_preds[sample_idx])
                # If no class is predicted, then we cannot extract any
                # explanation.
                explanations.append(
                    {'class': -1, 'explanation': '', 'attention': [], })
                continue

            # Otherwise we can extract an explanation for each active class!
            # e.g. we expain positive classifications.
            #print('A', sample_idx, active_classes)
            for target_class in active_classes:
                minterm = []
                for concept_idx in range(len(concept_names)):
                    c_pred = c[sample_idx, concept_idx]
                    sign_attn = sign_attn_mask[sample_idx, concept_idx, target_class]
                    filter_attn = filter_attn_mask[sample_idx, concept_idx, target_class]

                    # A concept is relevant <-> the filter attention score is
                    # lower than the concept probability.
                    sign_terms = self.logic.iff_pair(sign_attn, c_pred)
                    # print('IDX', concept_idx, 'C', c_pred.numpy(), 'ST', sign_terms.numpy(), 'FA', filter_attn.numpy(), 'FT', self.logic.disj_pair(self.logic.neg(filter_attn), sign_terms).numpy())

                    # If we are here we know that the prediction is positive, then all of
                    # self.logic.disj_pair(self.logic.neg(filter_attn), sign_terms) > 0.5
                    # Here, we only care of entries for which the attention is on, the others
                    # got assigned the neutral element and had no effect.
                    if filter_attn > 0.5:
                        if sign_attn >= 0.5:
                            # Concept is relevant and the sign is positive.
                            if mode == 'exact':
                                minterm.append(f'{sign_terms:.3f} ({concept_names[concept_idx]})')
                            else:
                                minterm.append(f'{concept_names[concept_idx]}')
                        else:
                            # Concept is relevant and the sign is negative.
                            if mode == 'exact':
                                minterm.append(f'{sign_terms:.3f} (~{concept_names[concept_idx]})')
                            else:
                                minterm.append(f'~{concept_names[concept_idx]}')

                # add explanation to list
                target_class_name = class_names[target_class]
                minterm = ' & '.join(minterm)
                all_class_explanations[target_class_name].append(minterm)
                explanations.append({
                    'sample-id': sample_idx,
                    'class': target_class_name,
                    'explanation': minterm,
                })

        if mode == 'global':
            # count most frequent explanations for each class
            explanations = []
            for class_id, class_explanations in all_class_explanations.items():
                explanation_count = Counter(class_explanations)
                for explanation, count in explanation_count.items():
                    explanations.append({
                        'class': class_id,
                        'explanation': explanation,
                        'count': count,
                    })

        return explanations
