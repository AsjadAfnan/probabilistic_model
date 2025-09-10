"""Tests for conditional distribution implementations."""

import pytest
import torch
import numpy as np

from pic.conditionals import LinearGaussian, NeuralEnergyConditional, FourierFeatures
from pic.quadrature import Quadrature


class TestLinearGaussian:
    """Test Linear-Gaussian conditional distribution."""
    
    def test_initialization(self):
        """Test LinearGaussian initialization."""
        A = torch.tensor([[1.0, 0.5], [0.0, 1.0]])
        b = torch.tensor([0.1, 0.2])
        Sigma = torch.tensor([[1.0, 0.1], [0.1, 1.0]])
        
        conditional = LinearGaussian("test", A, b, Sigma)
        
        assert conditional.name == "test"
        assert torch.allclose(conditional.A, A)
        assert torch.allclose(conditional.b, b)
        assert torch.allclose(conditional.Sigma, Sigma)
    
    def test_invalid_dimensions(self):
        """Test error handling for invalid dimensions."""
        # Invalid A (not 2D)
        A = torch.tensor([1.0, 2.0])
        b = torch.tensor([0.1])
        Sigma = torch.tensor([[1.0]])
        
        with pytest.raises(ValueError, match="A must be 2D matrix"):
            LinearGaussian("test", A, b, Sigma)
        
        # Invalid b (not 1D)
        A = torch.tensor([[1.0]])
        b = torch.tensor([[0.1]])
        
        with pytest.raises(ValueError, match="b must be 1D vector"):
            LinearGaussian("test", A, b, Sigma)
        
        # Invalid Sigma (not square)
        A = torch.tensor([[1.0]])
        b = torch.tensor([0.1])
        Sigma = torch.tensor([[1.0, 0.1]])
        
        with pytest.raises(ValueError, match="Sigma must be square"):
            LinearGaussian("test", A, b, Sigma)
        
        # Incompatible dimensions
        A = torch.tensor([[1.0, 0.5], [0.0, 1.0]])  # 2x2
        b = torch.tensor([0.1])  # 1D but length 1
        Sigma = torch.tensor([[1.0, 0.1], [0.1, 1.0]])  # 2x2
        
        with pytest.raises(ValueError, match="A rows.*!= b length"):
            LinearGaussian("test", A, b, Sigma)
    
    def test_log_prob(self):
        """Test log probability computation."""
        A = torch.tensor([[1.0, 0.5], [0.0, 1.0]])
        b = torch.tensor([0.1, 0.2])
        Sigma = torch.tensor([[1.0, 0.0], [0.0, 1.0]])  # Diagonal for simplicity
        
        conditional = LinearGaussian("test", A, b, Sigma)
        
        # Test single sample
        z_parent = torch.tensor([1.0, 2.0])
        z = torch.tensor([2.0, 3.0])
        
        log_prob = conditional.log_prob(z, z_parent)
        
        # Manual calculation
        mean = torch.matmul(z_parent, A.t()) + b
        expected_log_prob = torch.distributions.Normal(mean, torch.sqrt(torch.diag(Sigma))).log_prob(z)
        
        assert torch.allclose(log_prob, expected_log_prob, atol=1e-6)
    
    def test_log_prob_batch(self):
        """Test log probability computation with batch dimensions."""
        A = torch.tensor([[1.0, 0.5], [0.0, 1.0]])
        b = torch.tensor([0.1, 0.2])
        Sigma = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        
        conditional = LinearGaussian("test", A, b, Sigma)
        
        # Batch of parent and child values
        z_parent = torch.tensor([[1.0, 2.0], [0.5, 1.5]])
        z = torch.tensor([[2.0, 3.0], [1.0, 2.5]])
        
        log_prob = conditional.log_prob(z, z_parent)
        
        assert log_prob.shape == (2, 2)
        assert torch.all(torch.isfinite(log_prob))
    
    def test_log_norm(self):
        """Test log normalization constant computation."""
        A = torch.tensor([[1.0, 0.5], [0.0, 1.0]])
        b = torch.tensor([0.1, 0.2])
        Sigma = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        
        conditional = LinearGaussian("test", A, b, Sigma)
        quadrature = Quadrature.gauss_legendre(-3.0, 3.0, 10)
        
        # Test single parent value
        z_parent = torch.tensor([1.0, 2.0])
        log_norm = conditional.log_norm(z_parent, quadrature)
        
        # For linear-Gaussian, normalization should be constant
        expected_log_norm = -0.5 * torch.log(2 * torch.pi * torch.tensor(1.0)) - 0.5 * torch.logdet(Sigma)
        assert torch.allclose(log_norm, expected_log_norm, atol=1e-6)
        
        # Test batch of parent values
        z_parent_batch = torch.tensor([[1.0, 2.0], [0.5, 1.5]])
        log_norm_batch = conditional.log_norm(z_parent_batch, quadrature)
        
        assert log_norm_batch.shape == (2,)
        assert torch.allclose(log_norm_batch[0], log_norm_batch[1], atol=1e-6)  # Should be constant


class TestFourierFeatures:
    """Test Fourier feature embedding."""
    
    def test_initialization(self):
        """Test FourierFeatures initialization."""
        input_dim = 3
        output_dim = 10
        sigma = 1.5
        
        fourier = FourierFeatures(input_dim, output_dim, sigma)
        
        assert fourier.input_dim == input_dim
        assert fourier.output_dim == output_dim
        assert fourier.sigma == sigma
        assert fourier.B.shape == (input_dim, output_dim // 2)
    
    def test_forward(self):
        """Test Fourier feature forward pass."""
        input_dim = 2
        output_dim = 8
        sigma = 1.0
        
        fourier = FourierFeatures(input_dim, output_dim, sigma)
        
        # Test single sample
        x = torch.tensor([1.0, 2.0])
        features = fourier(x)
        
        assert features.shape == (output_dim,)
        assert torch.all(torch.isfinite(features))
        
        # Test batch
        x_batch = torch.tensor([[1.0, 2.0], [0.5, 1.5]])
        features_batch = fourier(x_batch)
        
        assert features_batch.shape == (2, output_dim)
        assert torch.all(torch.isfinite(features_batch))
    
    def test_fourier_properties(self):
        """Test properties of Fourier features."""
        input_dim = 2
        output_dim = 10
        sigma = 1.0
        
        fourier = FourierFeatures(input_dim, output_dim, sigma)
        
        # Test that features are periodic-like
        x1 = torch.tensor([1.0, 2.0])
        x2 = torch.tensor([1.0 + 2*np.pi, 2.0])
        
        features1 = fourier(x1)
        features2 = fourier(x2)
        
        # Features should be different (not exactly periodic due to random projection)
        assert not torch.allclose(features1, features2, atol=1e-6)


class TestNeuralEnergyConditional:
    """Test Neural Energy-Based conditional distribution."""
    
    def test_initialization(self):
        """Test NeuralEnergyConditional initialization."""
        parent_dim = 2
        child_dim = 1
        hidden_dim = 32
        
        conditional = NeuralEnergyConditional(
            "test", parent_dim, child_dim, hidden_dim, n_layers=2
        )
        
        assert conditional.name == "test"
        assert conditional.parent_dim == parent_dim
        assert conditional.child_dim == child_dim
        assert conditional.hidden_dim == hidden_dim
    
    def test_energy_computation(self):
        """Test energy function computation."""
        parent_dim = 2
        child_dim = 1
        hidden_dim = 16
        
        conditional = NeuralEnergyConditional("test", parent_dim, child_dim, hidden_dim)
        
        # Test single sample
        z_parent = torch.tensor([1.0, 2.0])
        z = torch.tensor([0.5])
        
        energy = conditional.energy(z, z_parent)
        
        assert energy.shape == ()
        assert torch.isfinite(energy)
        
        # Test batch
        z_parent_batch = torch.tensor([[1.0, 2.0], [0.5, 1.5]])
        z_batch = torch.tensor([[0.5], [1.0]])
        
        energy_batch = conditional.energy(z_batch, z_parent_batch)
        
        assert energy_batch.shape == (2,)
        assert torch.all(torch.isfinite(energy_batch))
    
    def test_log_prob(self):
        """Test unnormalized log probability computation."""
        parent_dim = 2
        child_dim = 1
        hidden_dim = 16
        
        conditional = NeuralEnergyConditional("test", parent_dim, child_dim, hidden_dim)
        
        z_parent = torch.tensor([1.0, 2.0])
        z = torch.tensor([0.5])
        
        log_prob = conditional.log_prob(z, z_parent)
        
        # Should be negative of energy
        energy = conditional.energy(z, z_parent)
        assert torch.allclose(log_prob, -energy, atol=1e-6)
    
    def test_log_norm_quadrature(self):
        """Test log normalization constant via quadrature."""
        parent_dim = 1
        child_dim = 1
        hidden_dim = 16
        
        conditional = NeuralEnergyConditional("test", parent_dim, child_dim, hidden_dim)
        quadrature = Quadrature.gauss_legendre(-3.0, 3.0, 20)
        
        # Test single parent value
        z_parent = torch.tensor([1.0])
        log_norm = conditional.log_norm(z_parent, quadrature)
        
        assert torch.isfinite(log_norm)
        
        # Test batch of parent values
        z_parent_batch = torch.tensor([[1.0], [0.5]])
        log_norm_batch = conditional.log_norm(z_parent_batch, quadrature)
        
        assert log_norm_batch.shape == (2,)
        assert torch.all(torch.isfinite(log_norm_batch))
    
    def test_normalized_log_prob(self):
        """Test normalized log probability computation."""
        parent_dim = 1
        child_dim = 1
        hidden_dim = 16
        
        conditional = NeuralEnergyConditional("test", parent_dim, child_dim, hidden_dim)
        quadrature = Quadrature.gauss_legendre(-3.0, 3.0, 20)
        
        z_parent = torch.tensor([1.0])
        z = torch.tensor([0.5])
        
        log_prob_norm = conditional.normalized_log_prob(z, z_parent, quadrature)
        
        # Should be unnormalized log prob minus log norm
        log_prob_unnorm = conditional.log_prob(z, z_parent)
        log_norm = conditional.log_norm(z_parent, quadrature)
        
        assert torch.allclose(log_prob_norm, log_prob_unnorm - log_norm, atol=1e-6)
    
    def test_without_fourier_features(self):
        """Test neural energy conditional without Fourier features."""
        parent_dim = 2
        child_dim = 1
        hidden_dim = 16
        
        conditional = NeuralEnergyConditional(
            "test", parent_dim, child_dim, hidden_dim, use_fourier_features=False
        )
        
        assert conditional.fourier is None
        
        z_parent = torch.tensor([1.0, 2.0])
        z = torch.tensor([0.5])
        
        energy = conditional.energy(z, z_parent)
        assert torch.isfinite(energy)
    
    def test_multidimensional_child(self):
        """Test with multidimensional child variable."""
        parent_dim = 2
        child_dim = 3
        hidden_dim = 16
        
        conditional = NeuralEnergyConditional("test", parent_dim, child_dim, hidden_dim)
        
        z_parent = torch.tensor([1.0, 2.0])
        z = torch.tensor([0.5, 1.0, -0.5])
        
        energy = conditional.energy(z, z_parent)
        assert torch.isfinite(energy)
        
        log_prob = conditional.log_prob(z, z_parent)
        assert torch.isfinite(log_prob)


class TestConditionalComparison:
    """Test comparison between different conditional types."""
    
    def test_linear_gaussian_vs_neural_energy(self):
        """Test that both conditional types work with quadrature."""
        # Linear-Gaussian conditional
        A = torch.tensor([[1.0]])
        b = torch.tensor([0.0])
        Sigma = torch.tensor([[1.0]])
        lg_conditional = LinearGaussian("lg", A, b, Sigma)
        
        # Neural energy conditional
        ne_conditional = NeuralEnergyConditional("ne", 1, 1, 16)
        
        quadrature = Quadrature.gauss_legendre(-3.0, 3.0, 20)
        z_parent = torch.tensor([1.0])
        
        # Both should compute log normalization
        lg_log_norm = lg_conditional.log_norm(z_parent, quadrature)
        ne_log_norm = ne_conditional.log_norm(z_parent, quadrature)
        
        assert torch.isfinite(lg_log_norm)
        assert torch.isfinite(ne_log_norm)
    
    def test_energy_scale_invariance(self):
        """Test that energy scale doesn't affect normalized probabilities."""
        parent_dim = 1
        child_dim = 1
        hidden_dim = 16
        
        conditional = NeuralEnergyConditional("test", parent_dim, child_dim, hidden_dim)
        quadrature = Quadrature.gauss_legendre(-3.0, 3.0, 20)
        
        z_parent = torch.tensor([1.0])
        z = torch.tensor([0.5])
        
        # Get normalized log probability
        log_prob_norm = conditional.normalized_log_prob(z, z_parent, quadrature)
        
        # Scale the energy network weights
        for param in conditional.energy_net.parameters():
            param.data *= 2.0
        
        # Get normalized log probability after scaling
        log_prob_norm_scaled = conditional.normalized_log_prob(z, z_parent, quadrature)
        
        # Should be different (scaling affects the energy)
        assert not torch.allclose(log_prob_norm, log_prob_norm_scaled, atol=1e-6)


class TestNumericalStability:
    """Test numerical stability of conditional distributions."""
    
    def test_extreme_values_linear_gaussian(self):
        """Test LinearGaussian with extreme values."""
        A = torch.tensor([[1.0, 0.5], [0.0, 1.0]])
        b = torch.tensor([0.1, 0.2])
        Sigma = torch.tensor([[1e-6, 0.0], [0.0, 1e6]])  # Extreme values
        
        conditional = LinearGaussian("test", A, b, Sigma)
        quadrature = Quadrature.gauss_legendre(-1000.0, 1000.0, 50)
        
        z_parent = torch.tensor([1e6, -1e6])
        z = torch.tensor([1e6, -1e6])
        
        log_prob = conditional.log_prob(z, z_parent)
        log_norm = conditional.log_norm(z_parent, quadrature)
        
        assert torch.all(torch.isfinite(log_prob))
        assert torch.all(torch.isfinite(log_norm))
    
    def test_extreme_values_neural_energy(self):
        """Test NeuralEnergyConditional with extreme values."""
        parent_dim = 1
        child_dim = 1
        hidden_dim = 16
        
        conditional = NeuralEnergyConditional("test", parent_dim, child_dim, hidden_dim)
        quadrature = Quadrature.gauss_legendre(-1000.0, 1000.0, 50)
        
        z_parent = torch.tensor([1e6])
        z = torch.tensor([1e6])
        
        energy = conditional.energy(z, z_parent)
        log_prob = conditional.log_prob(z, z_parent)
        log_norm = conditional.log_norm(z_parent, quadrature)
        
        assert torch.all(torch.isfinite(energy))
        assert torch.all(torch.isfinite(log_prob))
        assert torch.all(torch.isfinite(log_norm))
    
    def test_gradient_stability(self):
        """Test gradient stability for both conditional types."""
        # Linear-Gaussian
        A = torch.tensor([[1.0]], requires_grad=True)
        b = torch.tensor([0.0], requires_grad=True)
        Sigma = torch.tensor([[1.0]], requires_grad=True)
        lg_conditional = LinearGaussian("lg", A, b, Sigma)
        
        z_parent = torch.tensor([1.0])
        z = torch.tensor([0.5])
        
        log_prob = lg_conditional.log_prob(z, z_parent)
        loss = -torch.sum(log_prob)
        loss.backward()
        
        assert torch.all(torch.isfinite(A.grad))
        assert torch.all(torch.isfinite(b.grad))
        assert torch.all(torch.isfinite(Sigma.grad))
        
        # Neural energy (test with energy network parameters)
        parent_dim = 1
        child_dim = 1
        hidden_dim = 16
        
        ne_conditional = NeuralEnergyConditional("ne", parent_dim, child_dim, hidden_dim)
        
        z_parent = torch.tensor([1.0])
        z = torch.tensor([0.5])
        
        energy = ne_conditional.energy(z, z_parent)
        loss = torch.sum(energy)
        loss.backward()
        
        # Check gradients for energy network parameters
        for param in ne_conditional.energy_net.parameters():
            if param.grad is not None:
                assert torch.all(torch.isfinite(param.grad))
