import numpy as np
try:
    import cupy as cp
    USE_GPU = True
except:
    print('GMM import cupy failed, so use CPU version!')
    USE_GPU = False


"""
    code tested no problem
"""

class GMM:
    def __init__(self, num_samples, sample_dim=512, max_num_samples=30, sample_merge_type='merge',
                 learning_rate=0.009, metrics='euclidean', use_gpu=False):
        self._num_samples = num_samples
        self.sample_dim = sample_dim
        self.num_samples = max_num_samples
        self.learning_rate = learning_rate
        self.sample_merge_type = sample_merge_type
        self.metrics = metrics

        self.use_gpu = use_gpu and USE_GPU
        if not self.use_gpu:
            self._distance_matrix = np.ones((num_samples, num_samples), dtype=np.float32) * np.inf
            self._gram_matrix = np.ones((num_samples, num_samples), dtype=np.float32) * np.inf
            self.prior_weights = np.zeros((num_samples, 1), dtype=np.float32)
        else:
            self._distance_matrix = cp.ones((num_samples, num_samples), dtype=cp.float32) * cp.inf
            self._gram_matrix = cp.ones((num_samples, num_samples), dtype=cp.float32) * cp.inf
            self.prior_weights = cp.zeros((num_samples, 1), dtype=cp.float32)
        # find the minimum allowed sample weight. samples are discarded if their weights become lower
        self.minimum_sample_weight = self.learning_rate * (1 - self.learning_rate)**(2*self.num_samples)


    def _find_gram_vector(self, samplesf, new_sample, num_training_samples):
        if self.use_gpu:
            xp = cp.get_array_module(samplesf)
        else:
            xp = np
        gram_vector = xp.inf * xp.ones((self.num_samples))
        if num_training_samples > 0:
            samplesf_ = samplesf[:num_training_samples, :] # [n, dim]
            new_sample_ = new_sample.flatten()
            if self.metrics == 'cosine':
                samplesf_ = samplesf_ / xp.linalg.norm(samplesf_, axis=1, keepdims=True)
                new_sample_ = new_sample_ / xp.linalg.norm(new_sample_)
            # ip = xp.real(2 * samplesf_.dot(xp.conj(new_sample_)))
            ip = 2 * samplesf_.dot(new_sample_)
            
            gram_vector[:num_training_samples] = ip
        return gram_vector

    def init_samples(self, num_samples=None):
        num_samples = num_samples if num_samples is not None else self.num_samples
        if not self.use_gpu:
            # samples = np.zeros((num_samples, self.sample_dim), dtype=np.complex64)
            samples = np.zeros((num_samples, self.sample_dim), dtype=np.float)
        else:
            # samples = cp.zeros((num_samples, self.sample_dim), dtype=cp.complex64)
            samples = np.zeros((num_samples, self.sample_dim), dtype=np.float)
        return samples

    def _merge_samples(self, sample1, sample2, w1, w2, sample_merge_type):

        if sample_merge_type == 'replace':
            merged_sample = sample1
        elif sample_merge_type == 'merge':
            alpha1 = w1 / (w1 + w2)
            alpha2 = 1 - alpha1
            merged_sample = alpha1 * sample1 + alpha2 * sample2
        return merged_sample

    def _update_distance_matrix(self, gram_vector, new_sample_norm, id1, id2, w1, w2):
        """
            update the distance matrix
        """
        if self.use_gpu:
            xp = cp.get_array_module(gram_vector)
        else:
            xp = np
        alpha1 = w1 / (w1 + w2)
        alpha2 = 1 - alpha1
        if id2 < 0:
            norm_id1 = self._gram_matrix[id1, id1]

            # udpate the gram matrix
            if alpha1 == 0:
                self._gram_matrix[:, id1] = gram_vector
                self._gram_matrix[id1, :] = self._gram_matrix[:, id1]
                self._gram_matrix[id1, id1] = new_sample_norm
            elif alpha2 == 0:
                # new sample is discard
                pass
            else:
                # new sample is merge with an existing sample
                self._gram_matrix[:, id1] = alpha1 * self._gram_matrix[:, id1] + alpha2 * gram_vector
                self._gram_matrix[id1, :] = self._gram_matrix[:, id1]
                self._gram_matrix[id1, id1] = alpha1 ** 2 * norm_id1 + alpha2 ** 2 * new_sample_norm + 2 * alpha1 * alpha2 * gram_vector[id1]

            # udpate distance matrix
            self._distance_matrix[:, id1] = xp.maximum(self._gram_matrix[id1, id1] + xp.diag(self._gram_matrix) - 2 * self._gram_matrix[:, id1], 0)
            # self._distance_matrix[:, id1][np.isnan(self._distance_matrix[:, id1])] = 0
            self._distance_matrix[id1, :] = self._distance_matrix[:, id1]
            self._distance_matrix[id1, id1] = xp.inf
        else:
            if alpha1 == 0 or alpha2 == 0:
                raise("Error!")

            norm_id1 = self._gram_matrix[id1, id1]
            norm_id2 = self._gram_matrix[id2, id2]
            ip_id1_id2 = self._gram_matrix[id1, id2]

            # handle the merge of existing samples
            self._gram_matrix[:, id1] = alpha1 * self._gram_matrix[:, id1] + alpha2 * self._gram_matrix[:, id2]
            # self._distance_matrix[:, id1][np.isnan(self._distance_matrix[:, id1])] = 0
            self._gram_matrix[id1, :] = self._gram_matrix[:, id1]
            self._gram_matrix[id1, id1] = alpha1 ** 2 * norm_id1 + alpha2 ** 2 * norm_id2 + 2 * alpha1 * alpha2 * ip_id1_id2
            gram_vector[id1] = alpha1 * gram_vector[id1] + alpha2 * gram_vector[id2]

            # handle the new sample
            self._gram_matrix[:, id2] = gram_vector
            self._gram_matrix[id2, :] = self._gram_matrix[:, id2]
            self._gram_matrix[id2, id2] = new_sample_norm

            # update the distance matrix
            self._distance_matrix[:, id1] = xp.maximum(self._gram_matrix[id1, id1] + xp.diag(self._gram_matrix) - 2 * self._gram_matrix[:, id1], 0)
            self._distance_matrix[id1, :] = self._distance_matrix[:, id1]
            self._distance_matrix[id1, id1] = xp.inf
            self._distance_matrix[:, id2] = xp.maximum(self._gram_matrix[id2, id2] + xp.diag(self._gram_matrix) - 2 * self._gram_matrix[:, id2], 0)
            self._distance_matrix[id2, :] = self._distance_matrix[:, id2]
            self._distance_matrix[id2, id2] = xp.inf

    def update_sample_space_model(self, samplesf, new_train_sample, num_training_samples):
        if self.use_gpu:
            xp = cp.get_array_module(samplesf)
        else:
            xp = np

        # find the inner product of the new sample with existing samples
        gram_vector = self._find_gram_vector(samplesf, new_train_sample, num_training_samples)

        # find the inner product of the new sample with existing samples
        # new_train_sample_norm = xp.real(2 * xp.vdot(new_train_sample.flatten(), new_train_sample.flatten()))
        new_train_sample_norm = 2 * xp.dot(new_train_sample.flatten(), new_train_sample.flatten())
        dist_vector = xp.maximum(new_train_sample_norm + xp.diag(self._gram_matrix) - 2 * gram_vector, 0)
        dist_vector[num_training_samples:] = xp.inf

        merged_sample = self.init_samples(num_samples=0)
        new_sample = self.init_samples(num_samples=0)
        merged_sample_id = -1
        new_sample_id = -1

        if num_training_samples == self.num_samples:
            min_sample_id = xp.argmin(self.prior_weights)
            min_sample_weight = self.prior_weights[min_sample_id]
            if min_sample_weight < self.minimum_sample_weight:
                # if any prior weight is less than the minimum allowed weight
                # replace the sample with the new sample
                # udpate distance matrix and the gram matrix
                self._update_distance_matrix(gram_vector, new_train_sample_norm, min_sample_id, -1, 0, 1)

                # normalize the prior weights so that the new sample gets weight as the learning rate
                self.prior_weights[min_sample_id] = 0
                self.prior_weights = self.prior_weights * (1 - self.learning_rate) / xp.sum(self.prior_weights)
                self.prior_weights[min_sample_id] = self.learning_rate

                # set the new sample and new sample position in the samplesf
                new_sample_id = min_sample_id
                new_sample = new_train_sample
            else:
                # if no sample has low enough prior weight, then we either merge the new sample with
                # an existing sample, or merge two of the existing samples and insert the new sample
                # in the vacated position
                closest_sample_to_new_sample = xp.argmin(dist_vector)
                new_sample_min_dist = dist_vector[closest_sample_to_new_sample]

                # find the closest pair amongst existing samples
                closest_existing_sample_idx = xp.argmin(self._distance_matrix.flatten())
                closest_existing_sample_pair = xp.unravel_index(closest_existing_sample_idx, self._distance_matrix.shape)
                existing_samples_min_dist = self._distance_matrix[closest_existing_sample_pair[0], closest_existing_sample_pair[1]]
                closest_existing_sample1, closest_existing_sample2 = closest_existing_sample_pair
                if closest_existing_sample1 == closest_existing_sample2:
                    raise("Score matrix diagnoal filled wrongly")

                if new_sample_min_dist < existing_samples_min_dist:
                    # if the min distance of the new sample to the existing samples is less than the
                    # min distance amongst any of the existing samples, we merge the new sample with
                    # the nearest existing sample

                    # renormalize prior weights
                    self.prior_weights = self.prior_weights * (1 - self.learning_rate)

                    # set the position of the merged sample
                    merged_sample_id = closest_sample_to_new_sample

                    # extract the existing sample the merge
                    existing_sample_to_merge = samplesf[merged_sample_id:merged_sample_id+1, :]

                    # merge the new_training_sample with existing sample
                    merged_sample = self._merge_samples(existing_sample_to_merge,
                                                      new_train_sample,
                                                      self.prior_weights[merged_sample_id],
                                                      self.learning_rate,
                                                      self.sample_merge_type)

                    # update distance matrix and the gram matrix
                    self._update_distance_matrix(gram_vector,
                                                new_train_sample_norm,
                                                merged_sample_id,
                                                -1,
                                                self.prior_weights[merged_sample_id, 0],
                                                self.learning_rate)

                    # udpate the prior weight of the merged sample
                    self.prior_weights[closest_sample_to_new_sample] = self.prior_weights[closest_sample_to_new_sample] + self.learning_rate

                else:
                    # if the min distance amongst any of the existing samples is less than the
                    # min distance of the new sample to the existing samples, we merge the nearest
                    # existing samples and insert the new sample in the vacated position

                    # renormalize prior weights
                    self.prior_weights = self.prior_weights * ( 1 - self.learning_rate)

                    if self.prior_weights[closest_existing_sample2] > self.prior_weights[closest_existing_sample1]:
                        tmp = closest_existing_sample1
                        closest_existing_sample1 = closest_existing_sample2
                        closest_existing_sample2 = tmp

                    sample_to_merge1 = samplesf[closest_existing_sample1:closest_existing_sample1+1, :]
                    sample_to_merge2 = samplesf[closest_existing_sample2:closest_existing_sample2+1, :]
                    # merge the existing closest samples
                    merged_sample = self._merge_samples(sample_to_merge1,
                                                      sample_to_merge2,
                                                      self.prior_weights[closest_existing_sample1],
                                                      self.prior_weights[closest_existing_sample2],
                                                      self.sample_merge_type)

                    # update distance matrix and the gram matrix
                    self._update_distance_matrix(gram_vector,
                                                new_train_sample_norm,
                                                closest_existing_sample1,
                                                closest_existing_sample2,
                                                self.prior_weights[closest_existing_sample1, 0],
                                                self.prior_weights[closest_existing_sample2, 0])

                    # update prior weights for the merged sample and the new sample
                    self.prior_weights[closest_existing_sample1] = self.prior_weights[closest_existing_sample1] + self.prior_weights[closest_existing_sample2]
                    self.prior_weights[closest_existing_sample2] = self.learning_rate

                    # set the mreged sample position and new sample position
                    merged_sample_id = closest_existing_sample1
                    new_sample_id = closest_existing_sample2

                    new_sample = new_train_sample
        else:
            # if the memory is not full, insert the new sample in the next empty location
            sample_position = num_training_samples

            # update the distance matrix and the gram matrix
            self._update_distance_matrix(gram_vector, new_train_sample_norm, sample_position, -1, 0, 1)

            # update the prior weight
            if sample_position == 0:
                self.prior_weights[sample_position] = 1
            else:
                self.prior_weights = self.prior_weights * (1 - self.learning_rate)
                self.prior_weights[sample_position] = self.learning_rate

            new_sample_id = sample_position
            new_sample = new_train_sample

        if abs(1 - xp.sum(self.prior_weights)) > 1e-5:
            raise("weights not properly udpated")

        return merged_sample, new_sample, merged_sample_id, new_sample_id
