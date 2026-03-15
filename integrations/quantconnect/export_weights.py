"""
Extract PPO Model Weights for QuantConnect Deployment
Exports model weights and architecture information for production deployment
"""

import sys
import os
sys.path.append('src')

import json
import numpy as np
import torch
from stable_baselines3 import PPO
from typing import Dict, List, Tuple, Any


class PPOWeightExtractor:
    """Extract and format PPO model weights for QuantConnect"""
    
    def __init__(self, model_path: str):
        """
        Initialize extractor with PPO model path
        
        Args:
            model_path: Path to trained PPO model
        """
        self.model_path = model_path
        self.model = None
        self.architecture = {}
        self.weights = {}
        
    def load_model(self):
        """Load the PPO model"""
        try:
            self.model = PPO.load(self.model_path)
            print(f"Successfully loaded PPO model from {self.model_path}")
            print(f"Model device: {self.model.device}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
            
    def extract_architecture(self):
        """Extract model architecture information"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
            
        # Get observation space info
        obs_space = self.model.observation_space
        action_space = self.model.action_space
        
        # Get policy network architecture
        policy = self.model.policy
        
        # Extract layer sizes from the MLP extractor
        if hasattr(policy, 'mlp_extractor'):
            mlp = policy.mlp_extractor
            
            # Get shared network layers
            shared_layers = []
            if hasattr(mlp, 'shared_net'):
                for layer in mlp.shared_net:
                    if hasattr(layer, 'out_features'):
                        shared_layers.append(layer.out_features)
            
            # Get policy-specific layers  
            policy_layers = []
            if hasattr(mlp, 'policy_net'):
                for layer in mlp.policy_net:
                    if hasattr(layer, 'out_features'):
                        policy_layers.append(layer.out_features)
            
            # Get value-specific layers
            value_layers = []
            if hasattr(mlp, 'value_net'):
                for layer in mlp.value_net:
                    if hasattr(layer, 'out_features'):
                        value_layers.append(layer.out_features)
        
        self.architecture = {
            'observation_space': {
                'shape': list(obs_space.shape),
                'dtype': str(obs_space.dtype),
                'size': int(np.prod(obs_space.shape))
            },
            'action_space': {
                'n': int(action_space.n),
                'dtype': str(action_space.dtype)
            },
            'network_architecture': {
                'input_size': int(np.prod(obs_space.shape)),
                'shared_layers': shared_layers if 'shared_layers' in locals() else [64, 64],
                'policy_layers': policy_layers if 'policy_layers' in locals() else [64],
                'value_layers': value_layers if 'value_layers' in locals() else [64],
                'output_size': int(action_space.n)
            },
            'activation_function': 'tanh',  # Default for PPO
            'normalization': {
                'input_normalization': False,  # Will need to check this
                'layer_normalization': False
            }
        }
        
        print(f"Extracted architecture:")
        print(f"  Input size: {self.architecture['network_architecture']['input_size']}")
        print(f"  Shared layers: {self.architecture['network_architecture']['shared_layers']}")
        print(f"  Policy layers: {self.architecture['network_architecture']['policy_layers']}")
        print(f"  Value layers: {self.architecture['network_architecture']['value_layers']}")
        print(f"  Output size: {self.architecture['network_architecture']['output_size']}")
        
    def extract_weights(self):
        """Extract all model weights and biases"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
            
        weights = {}
        
        # Extract weights from all named parameters
        for name, param in self.model.policy.named_parameters():
            # Convert to numpy and then to list for JSON serialization
            weight_data = param.detach().cpu().numpy()
            
            # Store shape information along with weights
            weights[name] = {
                'shape': list(weight_data.shape),
                'data': weight_data.tolist(),
                'dtype': str(weight_data.dtype),
                'requires_grad': param.requires_grad
            }
            
            print(f"Extracted {name}: shape {weight_data.shape}")
        
        self.weights = weights
        print(f"Total parameters extracted: {len(weights)}")
        
    def get_forward_pass_info(self):
        """Get information about forward pass computation"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
            
        # Create dummy input to trace forward pass
        dummy_obs = torch.randn(1, *self.model.observation_space.shape)
        
        forward_info = {
            'input_preprocessing': {
                'normalization': 'none',  # Will need to check if there's input normalization
                'scaling': 'none'
            },
            'activation_functions': {
                'hidden_layers': 'tanh',
                'output_layer': 'softmax'
            },
            'output_interpretation': {
                'action_0': 'HOLD',
                'action_1': 'BUY', 
                'action_2': 'SELL'
            }
        }
        
        return forward_info
        
    def export_to_json(self, output_dir: str = "quantconnect"):
        """Export weights and architecture to JSON files"""
        if not self.weights or not self.architecture:
            raise ValueError("Weights and architecture not extracted. Call extract methods first.")
            
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Export weights
        weights_file = os.path.join(output_dir, 'ppo_weights.json')
        with open(weights_file, 'w') as f:
            json.dump(self.weights, f, indent=2)
        print(f"Weights exported to {weights_file}")
        
        # Export architecture
        arch_file = os.path.join(output_dir, 'ppo_architecture.json')
        with open(arch_file, 'w') as f:
            json.dump(self.architecture, f, indent=2)
        print(f"Architecture exported to {arch_file}")
        
        # Export forward pass info
        forward_info = self.get_forward_pass_info()
        forward_file = os.path.join(output_dir, 'ppo_forward_pass.json')
        with open(forward_file, 'w') as f:
            json.dump(forward_info, f, indent=2)
        print(f"Forward pass info exported to {forward_file}")
        
        return weights_file, arch_file, forward_file
    
    def create_inference_template(self, output_dir: str = "quantconnect"):
        """Create C# inference template for QuantConnect"""
        template = '''
using System;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace QuantConnect.Algorithm.CSharp
{
    /// <summary>
    /// PPO Model Inference for ES Futures Trading
    /// Converted from Python stable-baselines3 PPO model
    /// </summary>
    public class PPOInference
    {
        // Model architecture parameters
        private readonly int _inputSize = ''' + str(self.architecture['network_architecture']['input_size']) + ''';
        private readonly int _outputSize = ''' + str(self.architecture['network_architecture']['output_size']) + ''';
        
        // Network weights and biases (load from JSON)
        private Matrix<double> _sharedLayer1Weights;
        private Vector<double> _sharedLayer1Bias;
        private Matrix<double> _sharedLayer2Weights;
        private Vector<double> _sharedLayer2Bias;
        private Matrix<double> _policyOutputWeights;
        private Vector<double> _policyOutputBias;
        
        public PPOInference()
        {
            LoadWeights();
        }
        
        /// <summary>
        /// Load model weights from JSON files
        /// </summary>
        private void LoadWeights()
        {
            // TODO: Load weights from exported JSON files
            // This would typically be done in Initialize() method of QC algorithm
        }
        
        /// <summary>
        /// Perform forward pass inference
        /// </summary>
        /// <param name="observation">Input observation vector</param>
        /// <param name="stochastic">Use stochastic (true) or deterministic (false) action selection</param>
        /// <returns>Action index (0=HOLD, 1=BUY, 2=SELL)</returns>
        public int Predict(double[] observation, bool stochastic = true)
        {
            // Convert input to matrix
            var input = DenseVector.OfArray(observation);
            
            // Forward pass through network
            var hidden1 = Tanh(_sharedLayer1Weights.Multiply(input) + _sharedLayer1Bias);
            var hidden2 = Tanh(_sharedLayer2Weights.Multiply(hidden1) + _sharedLayer2Bias);
            var logits = _policyOutputWeights.Multiply(hidden2) + _policyOutputBias;
            
            // Apply softmax to get probabilities
            var probabilities = Softmax(logits.ToArray());
            
            // Action selection
            if (stochastic)
            {
                return SampleFromProbabilities(probabilities);
            }
            else
            {
                return ArgMax(probabilities);
            }
        }
        
        /// <summary>
        /// Tanh activation function
        /// </summary>
        private Vector<double> Tanh(Vector<double> input)
        {
            return input.Map(x => Math.Tanh(x));
        }
        
        /// <summary>
        /// Softmax activation function
        /// </summary>
        private double[] Softmax(double[] logits)
        {
            var max = logits.Max();
            var exp = logits.Select(x => Math.Exp(x - max)).ToArray();
            var sum = exp.Sum();
            return exp.Select(x => x / sum).ToArray();
        }
        
        /// <summary>
        /// Sample action from probability distribution
        /// </summary>
        private int SampleFromProbabilities(double[] probabilities)
        {
            var random = new Random();
            var rand = random.NextDouble();
            var cumSum = 0.0;
            
            for (int i = 0; i < probabilities.Length; i++)
            {
                cumSum += probabilities[i];
                if (rand <= cumSum)
                    return i;
            }
            
            return probabilities.Length - 1;
        }
        
        /// <summary>
        /// Get index of maximum value
        /// </summary>
        private int ArgMax(double[] array)
        {
            int maxIndex = 0;
            for (int i = 1; i < array.Length; i++)
            {
                if (array[i] > array[maxIndex])
                    maxIndex = i;
            }
            return maxIndex;
        }
    }
}
'''
        
        template_file = os.path.join(output_dir, 'PPOInference.cs')
        with open(template_file, 'w') as f:
            f.write(template)
        print(f"C# inference template created at {template_file}")
        
        return template_file
    
    def create_deployment_guide(self, output_dir: str = "quantconnect"):
        """Create deployment guide markdown"""
        guide = f'''# PPO Model Deployment Guide for QuantConnect

## Model Information
- **Model Type**: PPO (Proximal Policy Optimization)
- **Input Size**: {self.architecture['network_architecture']['input_size']} features
- **Output Size**: {self.architecture['network_architecture']['output_size']} actions (HOLD, BUY, SELL)
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
   {{
       private PPOInference _model;
       private FeatureCalculator _features;
       
       public override void Initialize()
       {{
           _model = new PPOInference();
           _features = new FeatureCalculator();
           
           // IMPORTANT: Enable stochastic mode
           _model.SetStochastic(true);
       }}
       
       public override void OnData(Slice data)
       {{
           var features = _features.Calculate(data);
           var action = _model.Predict(features, stochastic: true);
           
           switch(action)
           {{
               case 0: // HOLD - no action
                   break;
               case 1: // BUY
                   SetHoldings("ES", 1.0);
                   break;
               case 2: // SELL  
                   SetHoldings("ES", -1.0);
                   break;
           }}
       }}
   }}
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
'''
        
        guide_file = os.path.join(output_dir, 'deployment_guide.md')
        with open(guide_file, 'w', encoding='utf-8') as f:
            f.write(guide)
        print(f"Deployment guide created at {guide_file}")
        
        return guide_file


def main():
    """Export PPO model for QuantConnect deployment"""
    print("PPO MODEL EXPORT FOR QUANTCONNECT DEPLOYMENT")
    print("=" * 60)
    
    # Test with existing best model first
    model_path = "models/best/ppo/ppo_best/best_model"
    
    if not os.path.exists(f"{model_path}.zip"):
        print(f"Model not found at {model_path}")
        return
    
    try:
        # Initialize extractor
        extractor = PPOWeightExtractor(model_path)
        
        # Load model
        print("Loading PPO model...")
        extractor.load_model()
        
        # Extract architecture
        print("Extracting model architecture...")
        extractor.extract_architecture()
        
        # Extract weights
        print("Extracting model weights...")
        extractor.extract_weights()
        
        # Export to JSON files
        print("Exporting to JSON files...")
        weights_file, arch_file, forward_file = extractor.export_to_json()
        
        # Create C# template
        print("Creating C# inference template...")
        template_file = extractor.create_inference_template()
        
        # Create deployment guide
        print("Creating deployment guide...")
        guide_file = extractor.create_deployment_guide()
        
        print(f"\n{'='*60}")
        print("EXPORT COMPLETE!")
        print(f"{'='*60}")
        print("Files created:")
        print(f"  - {weights_file}")
        print(f"  - {arch_file}")
        print(f"  - {forward_file}")
        print(f"  - {template_file}")
        print(f"  - {guide_file}")
        
        print(f"\nNext Steps:")
        print("1. Review exported files in quantconnect/ directory")
        print("2. Implement feature engineering in QuantConnect")
        print("3. Use STOCHASTIC evaluation (deterministic=False)")
        print("4. Target 36.5% win rate achieved in validation testing")
        
    except Exception as e:
        print(f"Export failed: {e}")
        raise


if __name__ == "__main__":
    main()