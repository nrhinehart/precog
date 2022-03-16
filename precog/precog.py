import torch

def get_yaw(coords):
    """
    Parameters
    ==========
    coords : torch.Tensor
        Coords of shape (N, *, 2, 2) where the last dimension is for (x,y) points 

    Returns
    =======
    torch.Tensor
        Yaws of shape (N, *) in radian between (-pi, +pi]
    """

def rotate_about_origin(coords, yaw):
    """
    Parameters
    ==========
    coords : torch.Tensor
        Coords of shape (M, 2) 
    yaws : torch.Tensor
        Yaws of shape (N, *) in radian between (-pi, +pi]

    Returns
    =======
    torch.Tensor
        Coords of shape (N, *, M, 2) of (x,y) points rotated by yaw about their origin
    """

class Phi(object):
    pass

class ESP(torch.nn):

    def __init__(self, B=8, A=5, T=20, Tpast=10):
        """
        Parameters
        ==========
        B : int
            Batch size
        A : int
            Number of agents
        T : int
            Prediction horizon
        Tpast : int
            Length of agent past
        """
        self.B = B
        self.A = A
        self.T = T
        self.Tpast = Tpast

        # latent distribution
        self.latent_dist = torch.distributions.normal.Normal(0., 1.)

        # whiskers
        # TODO: build whisker template
        self.whisker_template = None
        self.n_whisker_points = self.whisker_template.shape[0]

        # overhead CNN
        self.n_convnet_layers = 8
        convnet_layers = [
                torch.nn.Conv2d(2, 32, 3, stride=1, padding=2),
                torch.nn.ReLU()]
        for _ in range(self.n_convnet_layers):
            convlayer = torch.nn.Conv2d(32, 32, 3, stride=1, padding=2)
            convnet_layers.append(convlayer)
            convnet_layers.append(torch.nn.ReLU())
        convlayer = torch.nn.Conv2d(32, 8, 3, stride=1, padding=2)
        convnet_layers.append(convlayer)
        convnet_layers.append(torch.nn.ReLU())
        self.overhead_cnn = torch.nn.Sequential(*convnet_layers)

        # past RNN
        # get latent h_0 corresponding to present time in the sequence.
        # https://pytorch.org/docs/stable/generated/torch.nn.GRU.html
        self.past_rnn = torch.nn.GRU(input_size=2, hidden_size=128, batch_first=True)

        # social MLP
        mlp_layers = [
                torch.nn.Linear(2*A - 2, 200),
                torch.nn.Tanh(),
                torch.nn.Linear(200, 50),
                torch.nn.Identity()]
        self.social_mlp = torch.nn.Sequential(*mlp_layers)

        # future RNN
        self.future_rnn_h = None
        self.future_rnn = torch.nn.GRUCell(
                input_size=self.A*8 + self.n_whisker_points*8 + 50 + 128,
                hidden_size=50)

        # future MLP
        mlp_layers = [
                torch.nn.Linear(50, 200),
                torch.nn.Tanh(),
                torch.nn.Linear(200, 6),
                torch.nn.Identity()]
        self.future_mlp = torch.nn.Sequential(*mlp_layers)
    
    def get_overhead_h(self, phi):
        """Geat overhead feature grid.

        Return
        ======
        torch.Tensor
            overhead_h has shape (B, 8, W, W)
        """
        # phi.overhead has shape (B, W, W, C)
        # overhead has shape (B, C, W, W)
        overhead = phi.overhead.permute(0, 3, 1, 2)
        return self.overhead_cnn(overhead)
    
    def get_past_h(self, phi, K):
        """Get past features.

        Return
        ======
        torch.Tensor
            pasts_h has shape (B, A, K, 256)
        """
        # phi.pasts has shape (B, A, tau + 1, 2)
        #     Contains past positions normalized so ego vehicle is at origin
        pasts = phi.pasts
        # pasts has shape (A, B, tau + 1, 2)
        pasts = pasts.swapaxes(0, 1)
        pasts_h = []
        for a, agent_past in enumerate(pasts):
            # agent_pasts has shape (B, tau + 1, 2)
            agent_past_local = phi.world2local(agent_past, origin=a)
            # past_h has shape (B, 1, 128)
            _, past_h = self.past_rnn(agent_past_local)
            pasts_h.append(past_h)
        # pasts_h has shape (B, A, 128)
        pasts_h = torch.cat(pasts_h, dim=1)
        tmp_past_h = []
        for a in range(self.A):
            other_pasts_h = torch.cat([pasts_h[:, :a], pasts_h[:, a + 1:]], dim=1)
            # other_pasts_h has shape (B, 128)
            other_pasts_h = torch.sum(other_pasts_h, dim=1)
            # past_h has shape (B, 256)
            past_h = torch.cat([pasts_h[:, a], other_pasts_h], dim=2)
            tmp_past_h.append(past_h)
        pasts_h = tmp_past_h
        # torch.stack(pasts_h, dim=1) has shape (B, A, 256)
        return torch.tile(torch.stack(pasts_h, dim=1).unsqueeze(-2), (K, 1))
    
    def get_whisker_h(self, t, K, x_hist, overhead_h):
        """Get whisker features.

        Parameters
        ==========
        x_hist : torch.Tensor
            Matrix of car positions. Has shape (B, A, K, T' + 2, 2)
            where t <= T'.

        Return
        ======
        torch.Tensor
            whisker_h has shape (B, A, K, n_whisker_points*8)
        """
        # yaws has shape (B, A, K)
        yaws = get_yaw(x_hist[..., t:t + 1, :])
        # whisker_points has shape (B, A, K, n_whisker_points, 2)
        whisker_points = rotate_about_origin(self.whisker_template, yaws)
        whisker_points = whisker_points + x_hist[..., t + 1, :]
        # whisker_points has shape (B, A, K*n_whisker_points, 2)
        whisker_points = whisker_points.reshape((self.B, self.A, -1, 2))
        # whisker_h has shape (B, 8, A, K*n_whisker_points)
        whisker_h = torch.nn.functional.grid_sample(overhead_h, whisker_points)
        # whisker_h has shape (B, A, K, n_whisker_points, 8)
        whisker_h = whisker_h.reshape((self.B, 8, self.A, K, -1)) \
                .transpose((0, 2, 3, 4, 1))
        return whisker_h.reshape((self.B, self.A, K, -1))

    def get_interps_h(self, t, K, x_hist, overhead_h):
        """Get interpolations features.

        Return
        ======
        torch.Tensor
            interps_h has shape (B, A, K, 8*A)
        """
        # x_hist_grid has shape (B, A, K, t, 2)
        x_hist_grid = phi.world2grid(x_hist)
        # sample_points has shape (B, A, K, 2)
        sample_points = x_hist_grid[..., t + 1, :]
        # interp_h has shape (B, 8, A, K)
        interp_h = torch.nn.functional.grid_sample(overhead_h, sample_points)
        # interp_h has shape (B, K, A, 8)
        interp_h = interp_h.transpose((0, 3, 2, 1))
        # interp_h.reshape((self.B, K, -1)) has shape (B, K, A*8)
        return torch.tile(interp_h.reshape((self.B, K, -1)).unsqueeze(1), (self.A, 1, 1))

    def get_social_disps_h(self, t, phi, K, x_hist):
        """Get social (vehicle displacement) features.
        
        Return
        ======
        torch.Tensor
            social_disps_h has shape (B, A, K, 50)
            
        TODO: is have social_mlp consume arbitrary sized inputs beyond 2*A-2?
        """
        social_disps = []
        for a in range(self.A):
            # x_hist_local has shape (B, A, K, t, 2)
            x_hist_local = phi.world2local(x_hist, origin=a)
            # x_hist_local has shape (B, K, A, 2)
            x_hist_local = x_hist_local[..., t + 1, :].swapaxes(1, 2)

            # social_disp has shape (B, K, A-1, 2)
            social_disp = torch.cat((x_hist_local[..., :a, :],
                    x_hist_local[..., a + 1:, :]), dim=2) - x_hist_local[..., a, :]
            # social_disp has shape (B, K, 2*A-2)
            social_disp = social_disp.reshape((self.B, K, -1))
            social_disps.append(social_disp)
        # social_disps has shape (B, A, K, 2*A-2)
        social_disps = torch.stack(social_disps, dim=1)
        return self.social_mlp(social_disps)
    
    def get_h(self, K, *args):
        """Get concatenated feature vector.

        Returns
        =======
        torch.Tensor
            h has shape (B, A, K, 50)
        """
        # h has shape (B, A, K, feat_size)
        h = torch.cat(args, dim=-1)
        # h has shape (B*A*K, feat_size)
        h = h.reshape((self.B*self.A*K, -1))
        # h has shape (B*A*K, 50)
        self.future_rnn_h = self.future_rnn(h, self.future_rnn_h)
        return h.reshape((self.B, self.A, K, -1))

    def get_step_ingredients(self, K, x_hist, h):
        """Compute step x and its components m, xi, sigma (for step t) from feature vector.

        Returns
        =======
        torch.Tensor
            mu_t has shape (B, A, K, 2)
        torch.Tensor
            xi_t has shape (B, A, K, 2, 2)
        torch.Tensor
            sigma_t has shape (B, A, K, 2, 2)
        torch.Tensor
            x_t has shape (B, A, K, 2)
        """
        # s has shape (B, A, K, 6)
        s = self.future_mlp(h)
        # s has shape (B, A, K, 3, 2)
        s = s.reshape((self.B, self.A, K, -1, 2))
        # mu_t has shape (B, A, K, 2)
        mu_t = s[..., 0, :]
        # xi_t has shape (B, A, K, 2, 2)
        xi_t = s[..., 1:, :]
        # sigma_t has shape (B, A, K, 2, 2)
        sigma_t = torch.matrix_exp(xi_t + xi_t.swapaxes(-1, -2))
        return mu_t, xi_t, sigma_t

    def forward(self, phi, Z, K):
        """
        Parameters
        ==========
        Z : torch.Tensor
            Has size (B, A, K, T, 2).
        K : int
            The number of predictions
        phi : torch.Tensor
            Represents batched scenes

        Returns
        =======
        torch.Tensor
            x_hist has shape (B, A, K, T, 2)
        torch.Tensor
            mu_hist has shape (B, A, K, T, 2)
        torch.Tensor
            xi_hist has shape (B, A, K, T, 2, 2)
        torch.Tensor
            mu_hist has shape (B, A, K, T, 2, 2)

        TODO: implement phi.world2local, phi.world2grid, whiskers
        """
        # overhead_h has shape (B, 8, W, W)
        self.overhead_h = self.get_overhead_h(phi)
        # pasts_h has shape (B, A, K, 256)
        self.pasts_h = self.get_past_h(phi, K)
        # phi.pasts has shape (B, A, tau + 1, 2)
        #     Contains past positions normalized so ego vehicle is at origin
        pasts = phi.pasts
        # x_hist has shape (B, A, K, t, 2)
        #     we stack more vectors along axis 3
        #     pasts[..., -2:, 2] has shape (B, A, 2, 2)
        x_hist = torch.tile(pasts[..., -2:, 2].unsqueeze(-3), (K, 1, 1))
        mu_hist    = []
        xi_hist    = []
        sigma_hist = []
        for t in range(self.T):
            # whisker_h has shape (B, A, K, n_whisker_points*8)
            whisker_h = self.get_whisker_h(t, K, x_hist, overhead_h)
            # interps_h has shape (B, A, K, 8*A)
            interps_h = self.get_interps_h(t, K, x_hist, overhead_h)
            # social_disps_h has shape (B, A, K, 50)
            social_disps_h = self.get_social_disps_h(t, phi, K, x_hist)
            # h has shape (B, A, K, 50)
            h = self.get_h(K, interps_h, whisker_h, social_disps_h, pasts_h)
            # mu_t has shape (B, A, K, 2)
            # xi_t has shape (B, A, K, 2, 2)
            # sigma_t has shape (B, A, K, 2, 2)
            mu_t, xi_t, sigma_t = self.get_step_ingredients(K, x_hist, h)
            # Z_t has shape (B, A, K, 2)
            Z_t = Z[..., -1, :]
            # x_t has shape (B, A, K, 2)
            #     x_hist[..., ?, :] has shape (B, A, K, 2)
            x_t = torch.einsum("...ij,...j->...i", sigma_t, Z_t) \
                    + 2*x_hist[..., -1, :] - x_hist[..., -2, :] + mu_t
            mu_hist.append(mu_t)
            xi_hist.append(xi_t)
            sigma_hist.append(sigma_t)
            # x_hist has shape (B, A, K, t, 2)
            #     up to shape (B, A, K, T + 2, 2)
            x_hist = torch.cat((x_hist, x_t.unsqueeze(-2)), dim=-2)

        # x_hist has shape (B, A, K, T, 2)
        x_hist = x_hist[..., 2:, :]
        # mu_hist has shape (B, A, K, T, 2)
        mu_hist = torch.stack(mu_hist, dim=-2)
        # xi_hist has shape (B, A, K, T, 2, 2)
        xi_hist = torch.stack(xi_hist, dim=-3)
        # mu_hist has shape (B, A, K, T, 2, 2)
        sigma_hist = torch.stack(mu_hist, dim=-3)
        # z_hist has shape (B, A, K, T, 2)
        z_hist = Z
        return x_hist, mu_hist, xi_hist, sigma_hist, z_hist

    def sample(self, phi, K=12):
        shape = (self.B, self.A, K, self.T)
        Z = self.latent_dist.sample(shape)
        x_hist, mu_hist, xi_hist, sigma_hist, z_hist = self.forward(phi, Z, K)
        return x_hist

    def backward(self, phi, X, K):
        """
        Parameters
        ==========
        X : torch.Tensor
            Has size (B, A, K, T, 2).
        K : int
            The number of predictions
        phi : torch.Tensor
            Represents batched scenes

        Returns
        =======
        """
        # overhead_h has shape (B, 8, W, W)
        self.overhead_h = self.get_overhead_h(phi)
        # pasts_h has shape (B, A, K, 256)
        self.pasts_h = self.get_past_h(phi, K)
        # x_hist has shape (B, A, K, t, 2)
        #     we stack more vectors along axis 3
        #     pasts[..., -2:, 2] has shape (B, A, 2, 2)

        # phi.pasts has shape (B, A, tau + 1, 2)
        #     Contains past positions normalized so ego vehicle is at origin
        pasts = phi.pasts
        # pasts has shape (B, A, K, 2, 2)
        # pasts[..., -2:, :] has shape (B, A, 2, 2)
        pasts = torch.tile(pasts[..., -2:, :].unsqueeze(-3), (K, 1, 1))
        # x_hist has shape (B, A, K, T + 2, 2)
        x_hist = np.concatenate((pasts, X), dim=-2)
        z_hist = []
        mu_hist    = []
        xi_hist    = []
        sigma_hist = []
        for t in range(self.T):
            # whisker_h has shape (B, A, K, n_whisker_points*8)
            whisker_h = self.get_whisker_h(t, K, x_hist, overhead_h)
            # interps_h has shape (B, A, K, 8*A)
            interps_h = self.get_interps_h(t, K, x_hist, overhead_h)
            # social_disps_h has shape (B, A, K, 50)
            social_disps_h = self.get_social_disps_h(t, phi, K, x_hist)
            # h has shape (B, A, K, 50)
            h = self.get_h(K, interps_h, whisker_h, social_disps_h, pasts_h)
            # mu_t has shape (B, A, K, 2)
            # xi_t has shape (B, A, K, 2, 2)
            # sigma_t has shape (B, A, K, 2, 2)
            mu_t, xi_t, sigma_t = self.get_step_ingredients(K, x_hist, h)
            # inv_sigma_t has shape (B, A, K, 2, 2)
            inv_sigma_t = torch.matrix_exp(-xi_t - xi_t.swapaxes(-1, -2))
            
            x_t = torch.einsum("...ij,...j->...i", sigma_t, Z_t) \
                    + 2*x_hist[..., -1, :] - x_hist[..., -2, :] + mu_t

            u_t = x_hist[..., t + 1,:] + x_hist[..., t - 1, :] - 2*x_hist[..., t, :] - mu_t
            # z_t has shape (B, A, K, 2)
            z_t = torch.einsum("...ij,...j->...i", inv_sigma_t, u_t)
            mu_hist.append(mu_t)
            xi_hist.append(xi_t)
            sigma_hist.append(sigma_t)
            z_hist.append(z_t)

        # x_hist has shape (B, A, K, T, 2)
        x_hist = X
        # mu_hist has shape (B, A, K, T, 2)
        mu_hist = torch.stack(mu_hist, dim=-2)
        # xi_hist has shape (B, A, K, T, 2, 2)
        xi_hist = torch.stack(xi_hist, dim=-3)
        # mu_hist has shape (B, A, K, T, 2, 2)
        sigma_hist = torch.stack(mu_hist, dim=-3)
        # z_hist has shape (B, A, K, T, 2)
        z_hist = torch.stack(z_hist, dim=-2)
        return x_hist, mu_hist, xi_hist, sigma_hist, z_hist

    def log_prob(self, phi, X, K):
        x_hist, mu_hist, xi_hist, sigma_hist, z_hist = self.backward(phi, X, K)
        # log q(x | Phi) = log N(z = f_inv(x|Phi),0,I) - |det(df/dz | z=f_inv(x|Phi))|
        
        # has shape 
        log_det_sigma = torch.einsum("...ii", xi_hist + xi_hist.swapaxes(-2, -1))
        
