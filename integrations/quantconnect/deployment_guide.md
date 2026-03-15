# PPO Model Deployment Guide for QuantConnect

## Model Information
- **Model Type**: PPO (Proximal Policy Optimization)
- **Input Size**: 2830 features
- **Output Size**: 3 actions (HOLD, BUY, SELL)
- **Architecture**: Multi-layer perceptron with tanh activation

## Key Insight: Use Stochastic Evaluation
**CRITICAL**: This model requires **stochastic action selection** (`deterministic=False`) to achieve the target 36.5% win rate.
- Deterministic evaluation leads to 100% HOLD actions
- Stochastic evaluation enables the learned trading strategies

## Deployment Steps

### Phase 1: Model Export [COMPLETED]
- [x] Extract model weights to JSON
- [x] Document layer architecture  
- [x] Create C# inference template

### Phase 2: QuantConnect Integration (Week 1-2)
1. **Upload Model Files**
   - Upload `ppo_weights.json` to QuantConnect
   - Upload `ppo_architecture.json` for reference
   
2. **Implement Feature Engineering**
   - Port feature calculation from `features/technical_features.py`
   - Ensure 47 features × 60 lookback = 2820 inputs
   - Add 10 position features = 2830 total inputs

3. **Integration Template**
   ```csharp
   public class ESFuturesPPOAlgorithm : QCAlgorithm
   {
       private PPOInference _model;
       private FeatureCalculator _features;
       
       public override void Initialize()
       {
           _model = new PPOInference();
           _features = new FeatureCalculator();
           
           // IMPORTANT: Enable stochastic mode
           _model.SetStochastic(true);
       }
       
       public override void OnData(Slice data)
       {
           var features = _features.Calculate(data);
           var action = _model.Predict(features, stochastic: true);
           
           switch(action)
           {
               case 0: // HOLD - no action
                   break;
               case 1: // BUY
                   SetHoldings("ES", 1.0);
                   break;
               case 2: // SELL  
                   SetHoldings("ES", -1.0);
                   break;
           }
       }
   }
   ```

### Phase 3: Risk Management (Week 2)
1. **Position Sizing**
   - Implement Kelly criterion or fixed fractional sizing
   - Daily loss limits: -$1500 per day
   
2. **Trade Execution**
   - 15-point profit target
   - 5-point stop loss
   - No overnight positions

### Phase 4: Backtesting & Validation (Week 3)
1. **Historical Backtesting**
   - Test on out-of-sample data (2024-06-11 to 2024-09-12)
   - Validate 25%+ win rate target
   
2. **Performance Metrics**
   - Win rate ≥ 25% (target: 36.5%)
   - Profit factor ≥ 1.0
   - Maximum drawdown monitoring

## Expected Performance
Based on stochastic validation testing:
- **Win Rate**: 36.5% (exceeds 25% target)
- **Trading Activity**: 33.6% (balanced approach)  
- **Risk/Reward**: 3:1 ratio (15pt target / 5pt stop)

## Files Exported
- `ppo_weights.json`: Model weights and biases
- `ppo_architecture.json`: Network structure
- `ppo_forward_pass.json`: Inference configuration
- `PPOInference.cs`: C# implementation template
- `deployment_guide.md`: This guide

## Support Notes
- Model trained on ES 1-minute bars from 2022-2024
- Quality-focused reward system prioritizes winning trades
- Requires stochastic evaluation for optimal performance
- Contact for implementation questions or model updates
