
import contextlib
import logging
import pdb
import tensorflow as tf

import precog.interface as interface
import precog.utils.tensor_util as tensoru

log = logging.getLogger(__file__)

# def add_inner_dimension(arr, K):
#     return tf.tile(arr[:, None], (1, K) + (1,) * (tensoru.rank(arr) - 1))
 
class ESPJointTrajectoryBijectionMixin:
    def forward(self, Z, phi):
        """Implements the forward component of the bijection. Implemented in __CAR FRAMES__.

        :param Z: latent states
        :param phi: context
        :returns: 
        :rtype: 

        """
        r = tensoru.rank(Z)

        with contextlib.ExitStack() as stack:
            if getattr(self, 'debug_eager', False):
                tape = tf.GradientTape(persistent=True)
                stack.enter_context(tape)
                tape.watch(phi.S_past_car_frames)
                tape.watch(phi.overhead_features)
            assert(r in (4, 5))
            if r == 4:
                B, A, T, D = tensoru.shape(Z)
                S_0 = phi.S_past_car_frames[..., -1, :]
                S_m1 = phi.S_past_car_frames[..., -2, :]
                batch_shape = (B,)
            elif r == 5:
                B, K, A, T, D = tensoru.shape(Z)
                # (B, A, T, D) -> (B, K, A, T, D)
                S_0 = tensoru.expand_and_tile_axis(phi.S_past_car_frames[..., -1, :], N=K, axis=1)
                S_m1 = tensoru.expand_and_tile_axis(phi.S_past_car_frames[..., -2, :], N=K, axis=1)
                batch_shape = (B, K)
            else:
                raise ValueError("Unhandled rank of latents to warp.")

            S_history = [S_m1, S_0]

            Z_history = []
            m_history = []
            mu_history = []
            sigel_history = []
            sigma_history = []
            metadata_history = []

            phi.prepare(batch_shape)
            self._prepare(batch_shape, phi)
            for t_idx in range(T):
                m_t, sigel_t, sigma_t = self.step_generate(S_history, phi)
                mu_t = m_t + 2 * S_history[-1] - S_history[-2]

                # Z first, then compute S.
                Z_t = Z[..., t_idx, :]
                S_t = mu_t + tf.einsum('...ij,...j->...i', sigma_t,  Z_t)
                phi.update_frames(S_t_car_frames=S_t, S_tm1_car_frames=S_history[-1])

                m_history.append(m_t)
                mu_history.append(mu_t)
                sigel_history.append(sigel_t)
                sigma_history.append(sigma_t)
                S_history.append(S_t)
                Z_history.append(Z_t)
                metadata_history.append(self.current_metadata)

                if getattr(self, 'debug_eager', False): pdb.set_trace()
                
        roll = interface.ESPRollout(
            S_car_frames_list=S_history[2:],
            Z_list=Z_history,
            m_list=m_history,
            mu_list=mu_history,
            sigma_list=sigma_history,
            sigel_list=sigel_history,
            metadata_list=metadata_history,
            phi=phi)
        return roll

    def inverse(self, S, phi):
        """Implements the inverse component of the bijection. 

        :param S: output states. Must be in __CAR FRAMES__.
        :param phi: context
        :returns: 
        :rtype: 

        """
        r = tensoru.rank(S)

        assert(r in (4, 5))
        if r == 4:
            B, A, T, D = tensoru.shape(S)
            S_0 = phi.S_past_car_frames[..., -1, :]
            S_m1 = phi.S_past_car_frames[..., -2, :]
            batch_shape = (B,)
        elif r == 5:
            B, K, A, T, D = tensoru.shape(S)
            S_0 = tensoru.expand_and_tile_axis(phi.S_past_car_frames[..., -1, :], N=K, axis=1)
            S_m1 = tensoru.expand_and_tile_axis(phi.S_past_car_frames[..., -2, :], N=K, axis=1)            
            batch_shape = (B, K)
        else:
            raise ValueError("Unhandled rank of data to invert.")
        
        S_history = [S_m1, S_0]

        Z_history = []
        m_history = []
        mu_history = []
        sigel_history = []
        sigma_history = []
        metadata_history = []

        phi.prepare(batch_shape)
        self._prepare(batch_shape, phi)
        for t_idx in range(T):
            m_t, sigel_t, sigma_t = self.step_generate(S_history, phi)
            mu_t = m_t + 2 * S_history[-1] - S_history[-2]

            # S first, then compute Z.
            S_t = S[..., t_idx, :]
            # expm(X) expm(-X) = I -> (expm(X))^{-1} = expm(-X). Avoids computing any inverses explicitly.
            sigma_t_inv = tf.linalg.expm(-1 * sigel_t)
            Z_t = tf.einsum('...ij,...j->...i', sigma_t_inv, S_t - mu_t)
            phi.update_frames(S_t_car_frames=S_t, S_tm1_car_frames=S_history[-1])
            
            m_history.append(m_t)
            mu_history.append(mu_t)
            sigel_history.append(sigel_t)
            sigma_history.append(sigma_t)
            S_history.append(S_t)
            Z_history.append(Z_t)
            metadata_history.append(self.current_metadata)
            
        roll = interface.ESPRollout(
            S_car_frames_list=S_history[2:],
            Z_list=Z_history,
            m_list=m_history,
            mu_list=mu_history,
            sigma_list=sigma_history,
            sigel_list=sigel_history,
            metadata_list=metadata_history,
            phi=phi)
        return roll

    def eager_ensure_forward_bijection(self, S, phi, lenient=False):
        """

        :param S: In __CAR FRAMES__.
        :param phi: 
        :returns: 
        :rtype: 

        """
        
        roll_inv = self.inverse(S, phi)
        s_diff_1 = tf.abs(roll_inv.S_car_frames - S).numpy()
        assert(s_diff_1.max() < 1e-3)
        
        roll_inv_forward = self.forward(roll_inv.Z, phi)
        z_diff = tf.abs(roll_inv_forward.Z - roll_inv.Z).numpy()
        s_diff_2 = tf.abs(roll_inv_forward.S_car_frames - S).numpy()

        # m_diff = tf.abs(roll_inv_forward.m - roll_inv.m).numpy()
        # mu_diff = tf.abs(roll_inv_forward.mu - roll_inv.mu).numpy()

        if not lenient:
            assert(z_diff.max() < 1e-3)
            assert(s_diff_2.max() < 1e-3)
        else:
            if z_diff.max() >= 1e-3:
                log.warning("Z diff in bijection is large!")
            if s_diff_2.max() >= 1e-3:
                log.warning("S diff in bijection is large!")

    def eager_ensure_inverse_bijection(self, Z, phi, lenient=False):
        roll_fwd = self.forward(Z, phi)
        roll_fwd_inverse = self.inverse(roll_fwd.S_car_frames, phi)
        diff = tf.abs(roll_fwd_inverse.Z - Z).numpy()
        if not lenient:
            assert(diff.max() < 1e-3)
        else:
            if diff.max() >= 1e-3:
                log.warning("S diff in bijection is large!")

    def check_gradients(self, z):
        log.warning("No gradients to check (check_gradients has not been overridden)")
        
