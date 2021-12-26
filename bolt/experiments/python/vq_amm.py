#!/usr/bin/env python

import abc
import numpy as np

from . import vquantizers as vq
from . import amm

KEY_NLOOKUPS = 'nlookups'


class VQMatmul(amm.ApproxMatmul, abc.ABC):
    def __init__(self, ncodebooks, ncentroids=None):
        self.ncodebooks = ncodebooks
        self.ncentroids = (self._get_ncentroids() if ncentroids is None
                           else ncentroids)
        self.enc = self._create_encoder(ncodebooks)
        self.reset_for_new_task()

    @abc.abstractmethod
    def _create_encoder(self, ncodebooks):  # to be overriden by subclasses
        return vq.PQEncoder(ncodebooks=ncodebooks, ncentroids=self.ncentroids,
                            **self._get_encoder_kwargs())

    # @abc.abstractmethod
    def _get_ncentroids(self):
        pass

    @abc.abstractmethod
    def get_speed_metrics(self, A, B, fixedA=False, fixedB=False):
        pass

    def _get_encoder_kwargs(self):  # to be overriden by subclasses
        return {}

    def reset_for_new_task(self):
        self.A_enc = None
        self.luts = None

    def fit(self, A, B, Y=None):
        _, D = A.shape
        if D < self.ncodebooks:
            raise amm.InvalidParametersException(
                'D < C: {} < {}'.format(D, self.ncodebooks))
        self.enc.fit(A, B.T)

    def set_A(self, A):
        self.A_enc = self.enc.encode_X(A)

    def set_B(self, B):
        self.luts = self.enc.encode_Q(B.T)

    def __call__(self, A, B):
        if self.A_enc is None:
            self.set_A(A)
        if self.luts is None:
            self.set_B(B)
        return self.enc.dists_enc(self.A_enc, self.luts)

    def get_params(self):
        return {'ncodebooks': self.ncodebooks}


# ================================================================ PQ

class PQMatmul(VQMatmul):

    def _create_encoder(self, ncodebooks):  # to be overriden by subclasses
        return vq.PQEncoder(ncodebooks=ncodebooks, ncentroids=self.ncentroids,
                            **self._get_encoder_kwargs())

    def _get_ncentroids(self):
        return 256

    def get_speed_metrics(self, A, B, fixedA=False, fixedB=False):
        # data encoding and LUT costs
        nmuls = 0
        nmuls += 0 if fixedA else A.shape[0] * A.shape[1] * self.ncentroids
        nmuls += 0 if fixedB else B.shape[0] * B.shape[1] * self.ncentroids
        nlookups = A.shape[0] * B.shape[1] * self.ncodebooks
        return {amm.KEY_NMULTIPLIES: nmuls, KEY_NLOOKUPS: nlookups}


# ================================================================ OPQ

class OPQMatmul(PQMatmul):

    def _get_encoder_kwargs(self):
        return dict(preproc='OPQ')

    def get_speed_metrics(self, A, B, fixedA=False, fixedB=False):
        metrics = super().get_speed_metrics(A, B, fixedA=fixedA, fixedB=fixedB)
        rot_nmuls = A.shape[0] * A.shape[1] * A.shape[1]  # OPQ rotation cost
        metrics[amm.KEY_NMULTIPLIES] += rot_nmuls
        return metrics


# ================================================================ Bolt

class BoltMatmul(PQMatmul):

    # def __init__(self, ncodebooks):
    #     self.ncodebooks = ncodebooks
    #     self.ncentroids = 16
    #     self.enc = self._create_encoder(self.ncodebooks)
    #     self._reset()

    def _get_ncentroids(self):
        return 16

    def _create_encoder(self, ncodebooks):
        return vq.PQEncoder(ncodebooks=ncodebooks, ncentroids=self.ncentroids,
                            quantize_lut=True,
                            # quantize_lut=False,
                            # accumulate_how='mean',
                            accumulate_how='sum',
                            upcast_every=-1,
                            # upcast_every=2,
                            # upcast_every=4,
                            # upcast_every=256,  # fine as long as using mean
                            # TODO set quantize_lut=True after debug
                            **self._get_encoder_kwargs())


class GEHTBoltMatmul_CovTopk(BoltMatmul):

    def _get_encoder_kwargs(self):
        return dict(
            preproc='GEHT', sample_how='deterministic', stats_mat='cov')


class GEHTBoltMatmul_CovSamp(BoltMatmul):

    def _get_encoder_kwargs(self):
        return dict(
            preproc='GEHT', sample_how='importance', stats_mat='cov')


class GEHTBoltMatmul_CorrTopk(BoltMatmul):

    def _get_encoder_kwargs(self):
        return dict(
            preproc='GEHT', sample_how='deterministic', stats_mat='corr')


class GEHTBoltMatmul_CorrSamp(BoltMatmul):

    def _get_encoder_kwargs(self):
        return dict(
            preproc='GEHT', sample_how='importance', stats_mat='corr')


class BoltSplits(BoltMatmul):

    def _get_encoder_kwargs(self):
        return dict(
            preproc='PQ', encode_algo='splits')

    def get_speed_metrics(self, A, B, fixedA=False, fixedB=False):
        metrics = super().get_speed_metrics(A, B, fixedA=fixedA, fixedB=fixedB)
        nmuls = 0
        nmuls += 0 if fixedB else B.shape[0] * B.shape[1] * self.ncentroids
        metrics[amm.KEY_NMULTIPLIES] = nmuls
        return metrics


class BoltMultiSplits(BoltMatmul):

    def _get_encoder_kwargs(self):
        return dict(encode_algo='multisplits')

    def get_speed_metrics(self, A, B, fixedA=False, fixedB=False):
        metrics = super().get_speed_metrics(A, B, fixedA=fixedA, fixedB=fixedB)
        nmuls = 0
        nmuls += 0 if fixedB else B.shape[0] * B.shape[1] * self.ncentroids
        metrics[amm.KEY_NMULTIPLIES] = nmuls
        return metrics


class BoltPermMultiSplits(BoltMatmul):

    def _get_encoder_kwargs(self):
        return dict(preproc='GEHT', encode_algo='multisplits')

    def get_speed_metrics(self, A, B, fixedA=False, fixedB=False):
        metrics = super().get_speed_metrics(A, B, fixedA=fixedA, fixedB=fixedB)
        nmuls = 0
        nmuls += 0 if fixedB else B.shape[0] * B.shape[1] * self.ncentroids
        metrics[amm.KEY_NMULTIPLIES] = nmuls
        return metrics


class PQPerm(PQMatmul):

    def _get_encoder_kwargs(self):
        return dict(preproc='GEHT')

    def get_speed_metrics(self, A, B, fixedA=False, fixedB=False):
        metrics = super().get_speed_metrics(A, B, fixedA=fixedA, fixedB=fixedB)
        nmuls = 0
        nmuls += 0 if fixedB else B.shape[0] * B.shape[1] * self.ncentroids
        metrics[amm.KEY_NMULTIPLIES] = nmuls
        return metrics


class PQMultiSplits(PQMatmul):

    def __init__(self, ncodebooks, ncentroids=256):
        super().__init__(ncodebooks=ncodebooks, ncentroids=ncentroids)

    def _get_encoder_kwargs(self):
        return dict(encode_algo='multisplits')

    def get_params(self):
        return {'ncodebooks': self.ncodebooks, 'ncentroids': self.ncentroids}

    def get_speed_metrics(self, A, B, fixedA=False, fixedB=False):
        metrics = super().get_speed_metrics(A, B, fixedA=fixedA, fixedB=fixedB)
        nmuls = 0
        nmuls += 0 if fixedB else B.shape[0] * B.shape[1] * self.ncentroids
        metrics[amm.KEY_NMULTIPLIES] = nmuls
        return metrics


class PQPermMultiSplits(PQMatmul):

    def __init__(self, ncodebooks, ncentroids=256):
        super().__init__(ncodebooks=ncodebooks, ncentroids=ncentroids)

    def _get_encoder_kwargs(self):
        return dict(preproc='GEHT', encode_algo='multisplits')

    def get_params(self):
        return {'ncodebooks': self.ncodebooks, 'ncentroids': self.ncentroids}

    def get_speed_metrics(self, A, B, fixedA=False, fixedB=False):
        metrics = super().get_speed_metrics(A, B, fixedA=fixedA, fixedB=fixedB)
        nmuls = 0
        nmuls += 0 if fixedB else B.shape[0] * B.shape[1] * self.ncentroids
        metrics[amm.KEY_NMULTIPLIES] = nmuls
        return metrics


# ================================================================ Mithral

class OldMithralPQ(PQMatmul):

    # def _get_ncentroids(self):
    #     return 16

    def __init__(self, ncodebooks):
        super().__init__(ncodebooks=ncodebooks, ncentroids=16)

    def _create_encoder(self, ncodebooks):
        return vq.PQEncoder(ncodebooks=ncodebooks, ncentroids=self.ncentroids,
                            encode_algo='multisplits',
                            quantize_lut=True,
                            upcast_every=16,  # fine as long as using mean
                            accumulate_how='mean')

    def get_speed_metrics(self, A, B, fixedA=False, fixedB=False):
        N, D = A.shape
        D, M = B.shape
        # data encoding and LUT costs
        nmuls = 0
        nmuls += 0 if fixedA else N * D  # offset + scale before quantize
        nmuls += 0 if fixedB else M * self.ncentroids * D
        # lookups given encoded data + luts
        nlookups = N * M * self.ncodebooks
        return {amm.KEY_NMULTIPLIES: nmuls, KEY_NLOOKUPS: nlookups}

class VingiloteMatmul(VQMatmul):

    def __init__(self, ncodebooks, lut_work_const=-1):
        self.lut_work_const = lut_work_const
        if (lut_work_const is not None) and (lut_work_const > 0) and (
                lut_work_const > ncodebooks):
            raise amm.InvalidParametersException(
                "lut_work_const > ncodebooks: {} > {}".format(
                    lut_work_const, ncodebooks))
        super().__init__(ncodebooks=ncodebooks, ncentroids=16)

    # def _get_ncentroids(self):
    #     return 16

    # def fit(self, A, B, Y=None):
    #     super().fit(self, A, B, Y=Y)

    def _create_encoder(self, ncodebooks):
        return vq.VingiloteEncoder(
            ncodebooks=ncodebooks, lut_work_const=self.lut_work_const)

    def get_params(self):
        return {'ncodebooks': self.ncodebooks,
                'lut_work_const': self.lut_work_const}

    def get_speed_metrics(self, A, B, fixedA=False, fixedB=False):
        N, D = A.shape
        D, M = B.shape
        # data encoding and LUT costs
        nmuls = 0
        nmuls += 0 if fixedA else N * D  # offset + scale before quantize
        nmuls_per_codebook_per_output = self.ncentroids * D
        nmuls_per_output = nmuls_per_codebook_per_output * self.ncodebooks
        nmuls += 0 if fixedB else nmuls_per_output * M
        # lookups given encoded data + luts
        nlookups = N * M * self.ncodebooks
        return {amm.KEY_NMULTIPLIES: nmuls, KEY_NLOOKUPS: nlookups}

    def set_B(self, B):
        self.luts, self.offset, self.scale = self.enc.encode_Q(B.T)

    def __call__(self, A, B):
        if self.A_enc is None:
            self.set_A(A)
            # ***
            # self.set_A(A) just does this:
            # self.A_enc = self.enc.encode_X(A)
            # ***

        if self.luts is None:
            # set_B:
            # sets self.luts, self.offset, self.scale
            # uses self.enc.centroids
            self.set_B(B)
        return self.enc.dists_enc(self.A_enc, self.luts,
                                  offset=self.offset, scale=self.scale)


class PlutoMatmul(VQMatmul):

    def __init__(
        self,
        ncodebooks,
        activation=None,
        nonzeros_heuristic="pq",
        objective="mse",
        accumulate_how="mean",
        lut_work_const=-1,
    ):
        self.activation = activation
        self.nonzeros_heuristic = nonzeros_heuristic
        self.objective = objective
        self.accumulate_how = accumulate_how
        self.lut_work_const = lut_work_const
        if (lut_work_const is not None) and (lut_work_const > 0) and (
                lut_work_const > ncodebooks):
            raise amm.InvalidParametersException(
                "lut_work_const > ncodebooks: {} > {}".format(
                    lut_work_const, ncodebooks))
        super().__init__(ncodebooks=ncodebooks, ncentroids=16)

    # def _get_ncentroids(self):
    #     return 16

    # def fit(self, A, B, Y=None):
    #     super().fit(self, A, B, Y=Y)

    def reset_for_new_task(self):
        self.A_enc = None
        # XXX - also self.B = None?
        # No! this must be called at each call of forward-pass

    def _create_encoder(self, ncodebooks):
        pluto_enc = vq.PlutoEncoder(
            ncodebooks=ncodebooks,
            activation=self.activation,
            nonzeros_heuristic=self.nonzeros_heuristic,
            objective=self.objective,
            accumulate_how=self.accumulate_how,
            lut_work_const=self.lut_work_const,
        )
        return pluto_enc

    def get_params(self):
        activation_str = 'None'
        if self.activation is not None:
            if hasattr(self.activation, '__name__'):
                activation_str = self.activation.__name__
            else:
                activation_str = str(self.activation)
        return {'ncodebooks': self.ncodebooks,
                'lut_work_const': self.lut_work_const,
                'activation': activation_str,
                'nonzeros_heuristic': self.nonzeros_heuristic,
                'objective': self.objective}

    def get_speed_metrics(self, A, B, fixedA=False, fixedB=False):
        N, D = A.shape
        D, M = B.shape
        # data encoding and LUT costs
        nmuls = 0
        nmuls += 0 if fixedA else N * D  # offset + scale before quantize
        nmuls_per_codebook_per_output = self.ncentroids * D
        nmuls_per_output = nmuls_per_codebook_per_output * self.ncodebooks
        nmuls += 0 if fixedB else nmuls_per_output * M
        # lookups given encoded data + luts
        nlookups = N * M * self.ncodebooks
        return {amm.KEY_NMULTIPLIES: nmuls, KEY_NLOOKUPS: nlookups}

    def fit(self, A, B, Y=None, output=None, bias=None):
        """

        Args:
            A: left of shape (N, D)
            B: right of shape (D, M)
            Y: desired A @ B if not None -- see ApproxMatmul -- ignored
            bias: shape broadcasts when adding A @ B + bias
        """
        # TODO use bias with nonlinearity
        _, D = A.shape
        if D < self.ncodebooks:
            raise amm.InvalidParametersException(
                'D < C: {} < {}'.format(D, self.ncodebooks))

        # self.enc.fit sets self.enc.splits_lists and self.enc.centroids
        # self.enc.fit also calls clusterize.learn_pluto
        self.luts, self.offset, self.scale = self.enc.fit(
            A, B.T, output=output, bias=bias)
        self.B = B

    def set_B(self, B):
        # TODO: use stddev and shape to verify with less memory
        assert np.array_equal(self.B, B)

    def __call__(self, A, B):
        if self.A_enc is None:
            # sets self.A_enc, uses self.enc.splits_lists and self.enc.offsets
            self.A_enc = self.enc.encode_X(A)

        if not np.array_equal(self.B, B):
            # TODO: use stddev and shape to verify with less memory
            raise ValueError("Pluto luts cannot be transferred to new B.")

        if self.luts is None:
            raise ValueError("Pluto luts must be pre-learned.")

        # MultiCodebookEncoder.dists_enc looks at:
        #   - quantize_lut
        #   - total_lut_offset
        #   - scale_by
        #   - upcast_every
        #   - accumulate_how
        #   - ncodebooks
        output = self.enc.dists_enc(
            self.A_enc,
            self.luts,
            offset=self.offset,
            scale=self.scale,
        )
        #rint(f"A.shape:{A.shape} B.shape:{B.shape} out.shape:{output.shape}")
        return output



class MithralMatmul(VQMatmul):

    def __init__(
        self,
        ncodebooks,
        nonzeros_heuristic="pq",
        lut_work_const=-1,
    ):
        self.nonzeros_heuristic = nonzeros_heuristic
        self.lut_work_const = lut_work_const
        if (lut_work_const is not None) and (lut_work_const > 0) and (
                lut_work_const > ncodebooks):
            raise amm.InvalidParametersException(
                "lut_work_const > ncodebooks: {} > {}".format(
                    lut_work_const, ncodebooks))
        super().__init__(ncodebooks=ncodebooks, ncentroids=16)

    # def _get_ncentroids(self):
    #     return 16

    # def fit(self, A, B, Y=None):
    #     super().fit(self, A, B, Y=Y)

    def _create_encoder(self, ncodebooks):
        mithral_enc = vq.MithralEncoder(
            ncodebooks=ncodebooks,
            nonzeros_heuristic=self.nonzeros_heuristic,
            lut_work_const=self.lut_work_const,
        )
        return mithral_enc

    def get_params(self):
        return {'ncodebooks': self.ncodebooks,
                'lut_work_const': self.lut_work_const}

    def get_speed_metrics(self, A, B, fixedA=False, fixedB=False):
        N, D = A.shape
        D, M = B.shape
        # data encoding and LUT costs
        nmuls = 0
        nmuls += 0 if fixedA else N * D  # offset + scale before quantize
        nmuls_per_codebook_per_output = self.ncentroids * D
        nmuls_per_output = nmuls_per_codebook_per_output * self.ncodebooks
        nmuls += 0 if fixedB else nmuls_per_output * M
        # lookups given encoded data + luts
        nlookups = N * M * self.ncodebooks
        return {amm.KEY_NMULTIPLIES: nmuls, KEY_NLOOKUPS: nlookups}

    def set_B(self, B):
        self.luts, self.offset, self.scale = self.enc.encode_Q(B.T)

    def __call__(self, A, B):
        if self.A_enc is None:
            self.set_A(A)
        if self.luts is None:
            self.set_B(B)
        return self.enc.dists_enc(self.A_enc, self.luts,
                                  offset=self.offset, scale=self.scale)


class MithralPQ(MithralMatmul):

    def __init__(self, ncodebooks):
        super().__init__(ncodebooks=ncodebooks, lut_work_const=1)
