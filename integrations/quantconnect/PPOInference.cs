
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
        private readonly int _inputSize = 2830;
        private readonly int _outputSize = 3;
        
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
