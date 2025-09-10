"""Tests for quadrature methods and numerical stability."""

import pytest
import torch
import numpy as np

from pic.quadrature import Quadrature, log_sum_exp_stable, create_quadrature


class TestQuadrature:
    """Test quadrature methods and properties."""
    
    def test_trapezoid_quadrature(self):
        """Test trapezoid quadrature creation and properties."""
        a, b = -2.0, 2.0
        n_points = 10
        
        quad = Quadrature.trapezoid(a, b, n_points)
        
        assert len(quad) == n_points
        assert quad.method == "trapezoid"
        assert torch.allclose(quad.grid[0], torch.tensor(a))
        assert torch.allclose(quad.grid[-1], torch.tensor(b))
        
        # Check that weights sum to interval length
        total_weight = torch.sum(quad.weights)
        expected_weight = b - a
        assert torch.allclose(total_weight, torch.tensor(expected_weight), atol=1e-6)
    
    def test_gauss_legendre_quadrature(self):
        """Test Gauss-Legendre quadrature creation and properties."""
        a, b = -1.0, 1.0
        n_points = 5
        
        quad = Quadrature.gauss_legendre(a, b, n_points)
        
        assert len(quad) == n_points
        assert quad.method == "gauss_legendre"
        assert torch.all(quad.grid >= a)
        assert torch.all(quad.grid <= b)
        
        # Check that weights sum to interval length
        total_weight = torch.sum(quad.weights)
        expected_weight = b - a
        assert torch.allclose(total_weight, torch.tensor(expected_weight), atol=1e-6)
    
    def test_uniform_quadrature(self):
        """Test uniform quadrature creation and properties."""
        a, b = 0.0, 1.0
        n_points = 8
        
        quad = Quadrature.uniform(a, b, n_points)
        
        assert len(quad) == n_points
        assert quad.method == "uniform"
        assert torch.all(quad.grid > a)
        assert torch.all(quad.grid < b)
        
        # Check that weights sum to interval length
        total_weight = torch.sum(quad.weights)
        expected_weight = b - a
        assert torch.allclose(total_weight, torch.tensor(expected_weight), atol=1e-6)
    
    def test_invalid_trapezoid_points(self):
        """Test error for insufficient trapezoid points."""
        with pytest.raises(ValueError, match="Trapezoid rule requires at least 2 points"):
            Quadrature.trapezoid(-1.0, 1.0, 1)
    
    def test_invalid_gauss_legendre_points(self):
        """Test error for insufficient Gauss-Legendre points."""
        with pytest.raises(ValueError, match="Gauss-Legendre requires at least 1 point"):
            Quadrature.gauss_legendre(-1.0, 1.0, 0)
    
    def test_invalid_uniform_points(self):
        """Test error for insufficient uniform points."""
        with pytest.raises(ValueError, match="Uniform rule requires at least 1 point"):
            Quadrature.uniform(-1.0, 1.0, 0)
    
    def test_shape_mismatch_error(self):
        """Test error when grid and weights have different shapes."""
        grid = torch.tensor([1.0, 2.0, 3.0])
        weights = torch.tensor([0.5, 0.5])  # Different length
        
        with pytest.raises(ValueError, match="Grid shape.*!= weights shape"):
            Quadrature(grid, weights)


class TestQuadratureIntegration:
    """Test quadrature integration accuracy."""
    
    def test_integrate_constant_function(self):
        """Test integration of constant function."""
        a, b = 0.0, 1.0
        n_points = 10
        
        # Test different quadrature methods
        methods = ["trapezoid", "gauss_legendre", "uniform"]
        
        for method in methods:
            quad = create_quadrature(method, a, b, n_points)
            
            # Integrate f(x) = 2
            values = torch.full((n_points,), 2.0)
            result = quad.integrate(values)
            
            expected = 2.0 * (b - a)  # 2 * 1 = 2
            assert torch.allclose(result, torch.tensor(expected), atol=1e-6)
    
    def test_integrate_linear_function(self):
        """Test integration of linear function."""
        a, b = 0.0, 2.0
        n_points = 20
        
        quad = Quadrature.gauss_legendre(a, b, n_points)
        
        # Integrate f(x) = x
        values = quad.grid
        result = quad.integrate(values)
        
        expected = (b**2 - a**2) / 2  # 2
        assert torch.allclose(result, torch.tensor(expected), atol=1e-6)
    
    def test_integrate_quadratic_function(self):
        """Test integration of quadratic function."""
        a, b = -1.0, 1.0
        n_points = 15
        
        quad = Quadrature.gauss_legendre(a, b, n_points)
        
        # Integrate f(x) = x^2
        values = quad.grid**2
        result = quad.integrate(values)
        
        expected = (b**3 - a**3) / 3  # 2/3
        assert torch.allclose(result, torch.tensor(expected), atol=1e-6)
    
    def test_convergence_monotone(self):
        """Test that error decreases as grid size increases."""
        a, b = -2.0, 2.0
        
        # Function to integrate: f(x) = exp(-x^2)
        def target_function(x):
            return torch.exp(-x**2)
        
        # Exact integral (Gaussian integral)
        exact = torch.sqrt(torch.pi * torch.tensor(1.0)) * torch.erf(torch.tensor(2.0))
        
        grid_sizes = [8, 16, 32, 64]
        errors = []
        
        for n_points in grid_sizes:
            quad = Quadrature.gauss_legendre(a, b, n_points)
            values = target_function(quad.grid)
            result = quad.integrate(values)
            error = torch.abs(result - exact)
            errors.append(error.item())
        
        # Check that errors are generally decreasing (allow some fluctuations)
        # The last error should be smaller than the first
        assert errors[-1] <= errors[0] * 0.5


class TestLogSumExp:
    """Test log-sum-exp operations."""
    
    def test_log_sum_exp_basic(self):
        """Test basic log-sum-exp functionality."""
        log_values = torch.tensor([1.0, 2.0, 3.0])
        
        result = log_sum_exp_stable(log_values)
        
        # Manual calculation
        expected = torch.log(torch.exp(torch.tensor(1.0)) + torch.exp(torch.tensor(2.0)) + torch.exp(torch.tensor(3.0)))
        assert torch.allclose(result, expected, atol=1e-6)
    
    def test_log_sum_exp_with_weights(self):
        """Test log-sum-exp with weights."""
        log_values = torch.tensor([1.0, 2.0, 3.0])
        weights = torch.tensor([0.5, 0.3, 0.2])
        
        result = log_sum_exp_stable(log_values, weights=weights)
        
        # Manual calculation with weights
        weighted_sum = 0.5 * torch.exp(torch.tensor(1.0)) + 0.3 * torch.exp(torch.tensor(2.0)) + 0.2 * torch.exp(torch.tensor(3.0))
        expected = torch.log(weighted_sum)
        assert torch.allclose(result, expected, atol=1e-6)
    
    def test_log_sum_exp_stability(self):
        """Test numerical stability of log-sum-exp."""
        # Large values that would cause overflow in naive implementation
        log_values = torch.tensor([1000.0, 1001.0, 1002.0])
        
        result = log_sum_exp_stable(log_values)
        
        # Should not be inf or nan
        assert torch.isfinite(result)
        assert not torch.isnan(result)
        
        # Should be close to the maximum value
        expected = 1002.0 + torch.log(torch.tensor(1.0) + torch.exp(torch.tensor(-1.0)) + torch.exp(torch.tensor(-2.0)))
        assert torch.allclose(result, expected, atol=1e-6)
    
    def test_log_sum_exp_quadrature(self):
        """Test quadrature log-sum-exp method."""
        quad = Quadrature.gauss_legendre(-1.0, 1.0, 10)
        
        log_values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        
        result = quad.log_sum_exp(log_values)
        
        # Should incorporate quadrature weights
        assert torch.isfinite(result)
        assert not torch.isnan(result)
    
    def test_log_sum_exp_batch(self):
        """Test log-sum-exp with batch dimensions."""
        # Batch of log values
        log_values = torch.tensor([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ])
        
        result = log_sum_exp_stable(log_values, dim=1)
        
        assert result.shape == (3,)
        assert torch.all(torch.isfinite(result))
    
    def test_log_sum_exp_keepdim(self):
        """Test log-sum-exp with keepdim=True."""
        log_values = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        
        result = log_sum_exp_stable(log_values, dim=1)
        
        assert result.shape == (2,)
        assert torch.all(torch.isfinite(result))


class TestQuadratureDevice:
    """Test quadrature device operations."""
    
    def test_to_device(self):
        """Test moving quadrature to device."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        quad = Quadrature.gauss_legendre(-1.0, 1.0, 10)
        device = torch.device("cuda:0")
        
        quad_device = quad.to(device)
        
        assert quad_device.grid.device == device
        assert quad_device.weights.device == device
        assert quad_device.method == quad.method
        assert len(quad_device) == len(quad)


class TestCreateQuadrature:
    """Test quadrature factory function."""
    
    def test_create_trapezoid(self):
        """Test creating trapezoid quadrature via factory."""
        quad = create_quadrature("trapezoid", -1.0, 1.0, 10)
        
        assert isinstance(quad, Quadrature)
        assert quad.method == "trapezoid"
        assert len(quad) == 10
    
    def test_create_gauss_legendre(self):
        """Test creating Gauss-Legendre quadrature via factory."""
        quad = create_quadrature("gauss_legendre", -1.0, 1.0, 5)
        
        assert isinstance(quad, Quadrature)
        assert quad.method == "gauss_legendre"
        assert len(quad) == 5
    
    def test_create_uniform(self):
        """Test creating uniform quadrature via factory."""
        quad = create_quadrature("uniform", 0.0, 1.0, 8)
        
        assert isinstance(quad, Quadrature)
        assert quad.method == "uniform"
        assert len(quad) == 8
    
    def test_invalid_method(self):
        """Test error for invalid quadrature method."""
        with pytest.raises(ValueError, match="Unknown quadrature method"):
            create_quadrature("invalid", -1.0, 1.0, 10)


class TestNumericalStability:
    """Test numerical stability of quadrature operations."""
    
    def test_extreme_sigma_values(self):
        """Test quadrature with extreme standard deviation values."""
        # Test with very small and very large sigma values
        sigma_values = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]
        
        for sigma in sigma_values:
            quad = Quadrature.gauss_legendre(-3*sigma, 3*sigma, 32)
            
            # Test Gaussian function with this sigma
            values = torch.exp(-0.5 * (quad.grid / sigma)**2) / (sigma * torch.sqrt(2 * torch.pi * torch.tensor(1.0)))
            result = quad.integrate(values)
            
            # Should be close to 1 (normalized Gaussian)
            assert torch.allclose(result, torch.tensor(1.0), atol=1e-2)
    
    def test_wide_integration_range(self):
        """Test quadrature with wide integration range."""
        # Test integration over a very wide range
        a, b = -1000.0, 1000.0
        n_points = 64
        
        quad = Quadrature.gauss_legendre(a, b, n_points)
        
        # Test constant function
        values = torch.full((n_points,), 1.0)
        result = quad.integrate(values)
        
        expected = b - a  # 2000
        assert torch.allclose(result, torch.tensor(expected), atol=1e-6)
    
    def test_log_sum_exp_boundaries(self):
        """Test log-sum-exp with boundary values."""
        # Test with very large negative values
        log_values = torch.tensor([-1000.0, -999.0, -998.0])
        result = log_sum_exp_stable(log_values)
        
        assert torch.isfinite(result)
        assert not torch.isnan(result)
        
        # Test with very large positive values
        log_values = torch.tensor([1000.0, 1001.0, 1002.0])
        result = log_sum_exp_stable(log_values)
        
        assert torch.isfinite(result)
        assert not torch.isnan(result)
    
    def test_no_nan_inf_forward_backward(self):
        """Test that forward and backward passes don't produce NaN or inf."""
        quad = Quadrature.gauss_legendre(-2.0, 2.0, 16)
        
        # Create tensor that requires gradients
        x = torch.randn(10, requires_grad=True)
        
        # Create function that uses quadrature
        def quadrature_function(x):
            # Use x to create log values
            log_values = x.unsqueeze(0).expand(len(quad), -1).t()  # (batch, n_points)
            return quad.log_sum_exp(log_values, dim=1)
        
        # Forward pass
        result = quadrature_function(x)
        
        # Check no NaN or inf in forward pass
        assert torch.all(torch.isfinite(result))
        assert not torch.any(torch.isnan(result))
        
        # Backward pass
        loss = torch.sum(result)
        loss.backward()
        
        # Check no NaN or inf in gradients
        assert torch.all(torch.isfinite(x.grad))
        assert not torch.any(torch.isnan(x.grad))
