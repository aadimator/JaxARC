#!/usr/bin/env python3
"""Verification script for the comprehensive demo setup."""

import sys
from pathlib import Path

def verify_imports():
    """Verify all required imports are available."""
    print("🔍 Verifying imports...")
    
    try:
        import jax
        import jax.numpy as jnp
        print(f"✅ JAX {jax.__version__} - Backend: {jax.default_backend()}")
    except ImportError as e:
        print(f"❌ JAX import failed: {e}")
        return False
    
    try:
        from jaxarc.envs import arc_reset, arc_step, create_bbox_config
        print("✅ JaxARC environment imports")
    except ImportError as e:
        print(f"❌ JaxARC environment imports failed: {e}")
        return False
    
    try:
        from jaxarc.parsers import MiniArcParser
        print("✅ JaxARC parser imports")
    except ImportError as e:
        print(f"❌ JaxARC parser imports failed: {e}")
        return False
    
    try:
        from jaxarc.utils.visualization import EnhancedVisualizer, VisualizationConfig
        print("✅ JaxARC visualization imports")
    except ImportError as e:
        print(f"❌ JaxARC visualization imports failed: {e}")
        return False
    
    try:
        from jaxarc.utils.config import get_config
        print("✅ JaxARC config imports")
    except ImportError as e:
        print(f"❌ JaxARC config imports failed: {e}")
        return False
    
    return True

def verify_configuration():
    """Verify configuration loading works."""
    print("\n🔧 Verifying configuration...")
    
    try:
        from jaxarc.utils.config import get_config
        
        # Test basic config
        config = get_config()
        print("✅ Basic configuration loaded")
        
        # Test MiniArc config
        config = get_config(overrides=["dataset=mini_arc"])
        print("✅ MiniArc configuration loaded")
        
        # Test bbox action config
        config = get_config(overrides=["action=bbox"])
        print("✅ Bbox action configuration loaded")
        
        # Test combined config
        config = get_config(overrides=["dataset=mini_arc", "action=bbox", "debug=on"])
        print("✅ Combined configuration loaded")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration verification failed: {e}")
        return False

def verify_jax_functionality():
    """Verify JAX functionality."""
    print("\n⚡ Verifying JAX functionality...")
    
    try:
        import jax
        import jax.numpy as jnp
        import jax.random as jr
        
        # Test basic JAX operations
        key = jr.PRNGKey(42)
        x = jr.normal(key, (10, 10))
        y = jnp.sum(x)
        print(f"✅ Basic JAX operations work")
        
        # Test JIT compilation
        @jax.jit
        def test_function(x):
            return jnp.sum(x ** 2)
        
        result = test_function(x)
        print(f"✅ JIT compilation works")
        
        # Test vmap
        batch_fn = jax.vmap(test_function)
        batch_x = jr.normal(key, (5, 10, 10))
        batch_result = batch_fn(batch_x)
        print(f"✅ Vectorization (vmap) works")
        
        return True
        
    except Exception as e:
        print(f"❌ JAX functionality verification failed: {e}")
        return False

def verify_environment():
    """Verify environment functionality."""
    print("\n🏃 Verifying environment functionality...")
    
    try:
        import jax.random as jr
        from jaxarc.envs import arc_reset, arc_step, create_bbox_config
        
        # Create config
        config = create_bbox_config(max_episode_steps=10)
        print("✅ Environment config created")
        
        # Test reset
        key = jr.PRNGKey(123)
        state, obs = arc_reset(key, config)
        print(f"✅ Environment reset works - obs shape: {obs.shape}")
        
        # Test step
        import jax.numpy as jnp
        action = {
            "bbox": jnp.array([0, 0, 2, 2], dtype=jnp.int32),
            "operation": jnp.array(1, dtype=jnp.int32)
        }
        
        new_state, new_obs, reward, done, info = arc_step(state, action, config)
        print(f"✅ Environment step works - reward: {reward:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Environment verification failed: {e}")
        return False

def check_dataset_availability():
    """Check if MiniArc dataset is available."""
    print("\n📁 Checking dataset availability...")
    
    # Common dataset locations
    possible_paths = [
        Path("data/raw/MiniARC"),
        Path("data/MiniARC"),
        Path("../data/raw/MiniARC"),
        Path("./MiniARC")
    ]
    
    for path in possible_paths:
        if path.exists():
            print(f"✅ MiniArc dataset found at: {path}")
            return True
    
    print("⚠️  MiniArc dataset not found at common locations:")
    for path in possible_paths:
        print(f"   {path} - {'exists' if path.exists() else 'not found'}")
    
    print("\n💡 To download MiniArc dataset:")
    print("   1. Visit: https://github.com/KSB21ST/MINI-ARC")
    print("   2. Download and extract to data/raw/MiniARC/")
    print("   3. Or use demo tasks (notebook will handle this)")
    
    return False

def main():
    """Run all verification checks."""
    print("🚀 JaxARC Comprehensive Demo Verification")
    print("=" * 50)
    
    checks = [
        ("Imports", verify_imports),
        ("Configuration", verify_configuration),
        ("JAX Functionality", verify_jax_functionality),
        ("Environment", verify_environment),
    ]
    
    results = []
    for name, check_fn in checks:
        try:
            result = check_fn()
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name} check crashed: {e}")
            results.append((name, False))
    
    # Dataset check (non-critical)
    dataset_available = check_dataset_availability()
    
    print("\n" + "=" * 50)
    print("📊 Verification Summary")
    print("=" * 50)
    
    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name:20} {status}")
        if not passed:
            all_passed = False
    
    dataset_status = "✅ AVAILABLE" if dataset_available else "⚠️  NOT FOUND"
    print(f"{'Dataset':20} {dataset_status}")
    
    if all_passed:
        print(f"\n🎉 All critical checks passed!")
        print(f"The comprehensive demo is ready to run.")
        if not dataset_available:
            print(f"Note: Demo will use built-in tasks if dataset is not available.")
    else:
        print(f"\n❌ Some checks failed. Please fix the issues before running the demo.")
        return 1
    
    print(f"\n🚀 To run the comprehensive demo:")
    print(f"   pixi run jupyter notebook jaxarc_comprehensive_demo.py")
    print(f"   # or")
    print(f"   pixi run python jaxarc_comprehensive_demo.py")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())