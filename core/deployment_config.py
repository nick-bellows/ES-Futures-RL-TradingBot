"""
Deployment Configuration for ES Futures RL Trading Bot

This file defines the difference between training and production parameters.
CRITICAL: Risk/reward ratios must remain identical between training and deployment.
"""

# =============================================================================
# CORE TRADE MECHANICS - NEVER CHANGE BETWEEN TRAINING AND DEPLOYMENT
# =============================================================================
# The agent must learn the EXACT risk/reward dynamics it will face in production

PRODUCTION_TRADE_CONFIG = {
    "stop_loss_points": 5.0,        # 🔒 LOCKED: 5-point stop loss
    "profit_target_points": 15.0,   # 🔒 LOCKED: 15-point target (3:1 ratio)
    "point_value": 50.0,            # 🔒 LOCKED: ES futures contract value
}

# These MUST be identical in both training and deployment
# If the agent learns 2:1 risk/reward, it will fail with 3:1 in production

# =============================================================================
# EXPLORATION LIMITS - DIFFERENT FOR TRAINING VS DEPLOYMENT
# =============================================================================

TRAINING_CONFIG = {
    # Core trade mechanics (SAME as production)
    **PRODUCTION_TRADE_CONFIG,
    
    # Exploration limits (RELAXED for learning)
    "daily_loss_limit": 1500,       # 🔄 2x production limit (more exploration)
    "max_daily_trades": 10,         # 🔄 2x production limit (more opportunities)
    
    # Training specific
    "initial_balance": 50000,
    "lookback_window": 60,
    "reward_scaling": 1.0,
}

DEPLOYMENT_CONFIG = {
    # Core trade mechanics (SAME as training)
    **PRODUCTION_TRADE_CONFIG,
    
    # Risk limits (STRICT for production)
    "daily_loss_limit": 750,        # 🛡️ Original strict limit
    "max_daily_trades": 5,          # 🛡️ Original conservative limit
    
    # Production specific
    "initial_balance": 50000,       # Real account balance
    "lookback_window": 60,
    "reward_scaling": 1.0,
    
    # Additional production safeguards
    "pre_trade_validation": True,
    "real_time_monitoring": True,
    "position_size_validation": True,
    "market_hours_enforcement": True,
}

# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_config_consistency():
    """Ensure core trade mechanics are identical between training and deployment"""
    
    training_core = {
        k: v for k, v in TRAINING_CONFIG.items() 
        if k in PRODUCTION_TRADE_CONFIG
    }
    
    deployment_core = {
        k: v for k, v in DEPLOYMENT_CONFIG.items() 
        if k in PRODUCTION_TRADE_CONFIG  
    }
    
    if training_core != deployment_core:
        raise ValueError(
            f"CRITICAL ERROR: Core trade mechanics differ between training and deployment!\n"
            f"Training: {training_core}\n"
            f"Deployment: {deployment_core}\n"
            f"These must be identical for the agent to work in production."
        )
    
    print("✅ Configuration validation passed: Core mechanics are consistent")
    return True

def get_training_env_params():
    """Get parameters for training environment"""
    return TRAINING_CONFIG

def get_deployment_env_params():
    """Get parameters for deployment environment"""
    return DEPLOYMENT_CONFIG

# =============================================================================
# USAGE EXAMPLES
# =============================================================================

if __name__ == "__main__":
    print("ES Futures RL Trading Bot - Configuration Validation")
    print("=" * 60)
    
    # Validate consistency
    validate_config_consistency()
    
    print("\nTRAINING CONFIGURATION:")
    for key, value in TRAINING_CONFIG.items():
        print(f"  {key}: {value}")
    
    print("\nDEPLOYMENT CONFIGURATION:")
    for key, value in DEPLOYMENT_CONFIG.items():
        print(f"  {key}: {value}")
    
    print("\nKEY DIFFERENCES:")
    print("  TRAINING: More trades (10), higher loss limit ($1500) for exploration")
    print("  DEPLOYMENT: Fewer trades (5), lower loss limit ($750) for safety")
    print("  IDENTICAL: 5-point stops, 15-point targets (3:1 ratio) for consistency")
    
    print("\n✅ Configuration is ready for both training and deployment!")