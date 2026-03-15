import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from tradovate_replay_backtester import TradovateReplayBacktester
import threading
import time

def run_single_replay_test(model_path, date, market_type, username, password):
    """Run a single replay test"""
    print(f"\n{'='*50}")
    print(f"Testing {date} - {market_type}")
    print(f"{'='*50}")
    
    try:
        backtester = TradovateReplayBacktester(model_path, date)
        backtester.connect_replay(username, password)
        
        # Run in a separate thread with timeout
        def run_test():
            backtester.run()
        
        test_thread = threading.Thread(target=run_test)
        test_thread.daemon = True
        test_thread.start()
        
        # Wait for completion (max 30 minutes for 10x speed)
        test_thread.join(timeout=1800)
        
        if test_thread.is_alive():
            print(f"Test for {date} timed out")
            return None
        
        # Collect results
        metrics = backtester.calculate_metrics()
        metrics['date'] = date
        metrics['market_type'] = market_type
        
        print(f"Results: Win Rate={metrics['win_rate']:.1%}, "
              f"P&L=${metrics['total_pnl']:.2f}, "
              f"Trades={metrics['num_trades']}")
        
        return metrics
        
    except Exception as e:
        print(f"Error testing {date}: {e}")
        return None

def run_replay_tests(model_path, username, password):
    """Test on different market conditions"""
    
    test_dates = [
        ('2024-01-15', 'Trending Up'),    # Strong trend day
        ('2024-02-28', 'Trending Down'),  # Sell-off day
        ('2024-03-15', 'Range Bound'),    # Choppy day
        ('2024-04-10', 'High Volatility'), # News day
        ('2024-05-03', 'Normal Day'),     # Typical day
        ('2024-06-12', 'Earnings Day'),   # Earnings volatility
        ('2024-07-25', 'Fed Day'),        # Fed announcement
        ('2024-08-14', 'Low Volume'),     # Summer trading
        ('2024-09-18', 'Options Expiry'), # OpEx day
        ('2024-10-31', 'Month End'),      # Month-end rebalancing
    ]
    
    results = []
    
    for date, market_type in test_dates:
        result = run_single_replay_test(model_path, date, market_type, username, password)
        if result:
            results.append(result)
        
        # Small delay between tests
        time.sleep(5)
    
    if not results:
        print("No successful test results")
        return pd.DataFrame()
    
    return pd.DataFrame(results)

def run_stress_tests(model_path, username, password):
    """Run stress tests on challenging market conditions"""
    
    stress_dates = [
        ('2024-01-03', 'First Trading Day'),
        ('2024-02-05', 'Market Crash Day'),
        ('2024-03-20', 'Fed Rate Decision'),
        ('2024-04-26', 'Earnings Season'),
        ('2024-05-31', 'Memorial Day'),
        ('2024-08-05', 'Flash Crash'),
        ('2024-10-07', 'Geopolitical Tension'),
        ('2024-11-05', 'Election Day'),
        ('2024-12-31', 'Year End'),
    ]
    
    print("\n" + "="*60)
    print("RUNNING STRESS TESTS")
    print("="*60)
    
    stress_results = []
    
    for date, event_type in stress_dates:
        result = run_single_replay_test(model_path, date, event_type, username, password)
        if result:
            stress_results.append(result)
        time.sleep(5)
    
    return pd.DataFrame(stress_results)

def run_parameter_sensitivity_tests(base_model_path, username, password):
    """Test different confidence thresholds"""
    
    confidence_levels = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    test_date = '2024-05-15'  # Use a standard day
    
    print("\n" + "="*60)
    print("RUNNING PARAMETER SENSITIVITY TESTS")
    print("="*60)
    
    sensitivity_results = []
    
    for confidence in confidence_levels:
        print(f"\nTesting confidence threshold: {confidence}")
        
        # Modify the backtester to use different confidence threshold
        backtester = TradovateReplayBacktester(base_model_path, test_date)
        
        # Override the confidence threshold
        original_process = backtester.process_trading_signal
        
        def modified_process():
            try:
                df = pd.DataFrame(list(backtester.price_history))
                features = backtester.calculate_features(df)
                
                obs = features.reshape(1, -1)
                action, _ = backtester.model.predict(obs, deterministic=False)
                
                import torch
                with torch.no_grad():
                    obs_tensor = backtester.model.policy.obs_to_tensor(obs)[0]
                    distribution = backtester.model.policy.get_distribution(obs_tensor)
                    probs = torch.softmax(distribution.distribution.logits, dim=-1)
                    conf = probs[0][action].item()
                
                if conf > confidence:  # Use the test confidence level
                    current_price = backtester.price_history[-1]['close']
                    backtester.execute_replay_trade(action, current_price, conf)
                    
            except Exception as e:
                print(f"Error in modified process: {e}")
        
        backtester.process_trading_signal = modified_process
        
        try:
            backtester.connect_replay(username, password)
            
            def run_test():
                backtester.run()
            
            test_thread = threading.Thread(target=run_test)
            test_thread.daemon = True
            test_thread.start()
            test_thread.join(timeout=1800)
            
            metrics = backtester.calculate_metrics()
            metrics['confidence_threshold'] = confidence
            metrics['date'] = test_date
            
            sensitivity_results.append(metrics)
            
            print(f"Confidence {confidence}: {metrics['num_trades']} trades, "
                  f"{metrics['win_rate']:.1%} win rate, ${metrics['total_pnl']:.2f} P&L")
            
        except Exception as e:
            print(f"Error testing confidence {confidence}: {e}")
        
        time.sleep(5)
    
    return pd.DataFrame(sensitivity_results)

def create_comprehensive_report(results_df, stress_df=None, sensitivity_df=None):
    """Generate comprehensive test report"""
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 15))
    
    # Main results - 2x2 grid
    ax1 = plt.subplot(3, 3, 1)
    ax2 = plt.subplot(3, 3, 2)
    ax3 = plt.subplot(3, 3, 3)
    ax4 = plt.subplot(3, 3, 4)
    
    # Win rate by market type
    if not results_df.empty:
        market_stats = results_df.groupby('market_type').agg({
            'win_rate': 'mean',
            'total_pnl': 'mean',
            'num_trades': 'mean'
        }).reset_index()
        
        bars = ax1.bar(range(len(market_stats)), market_stats['win_rate'])
        ax1.set_xticks(range(len(market_stats)))
        ax1.set_xticklabels(market_stats['market_type'], rotation=45, ha='right')
        ax1.axhline(y=0.25, color='r', linestyle='--', label='Break-even (25%)')
        ax1.set_title('Win Rate by Market Condition')
        ax1.set_ylabel('Win Rate')
        ax1.legend()
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.1%}', ha='center', va='bottom')
    
    # P&L distribution
    if not results_df.empty:
        ax2.hist(results_df['total_pnl'], bins=10, alpha=0.7, edgecolor='black')
        ax2.axvline(x=0, color='r', linestyle='--', label='Break-even')
        ax2.set_title('P&L Distribution')
        ax2.set_xlabel('Daily P&L ($)')
        ax2.set_ylabel('Frequency')
        ax2.legend()
    
    # Trade frequency vs Win rate scatter
    if not results_df.empty:
        scatter = ax3.scatter(results_df['num_trades'], results_df['win_rate'], 
                             c=results_df['total_pnl'], cmap='RdYlGn', s=100, alpha=0.7)
        ax3.set_xlabel('Number of Trades')
        ax3.set_ylabel('Win Rate')
        ax3.set_title('Trade Frequency vs Win Rate')
        plt.colorbar(scatter, ax=ax3, label='P&L ($)')
    
    # Cumulative P&L over time
    if not results_df.empty:
        results_sorted = results_df.sort_values('date')
        cumulative_pnl = results_sorted['total_pnl'].cumsum()
        ax4.plot(range(len(cumulative_pnl)), cumulative_pnl, marker='o', linewidth=2)
        ax4.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax4.set_title('Cumulative P&L Over Test Dates')
        ax4.set_xlabel('Test Number')
        ax4.set_ylabel('Cumulative P&L ($)')
        ax4.grid(True, alpha=0.3)
    
    # Stress test results
    if stress_df is not None and not stress_df.empty:
        ax5 = plt.subplot(3, 3, 5)
        stress_pnl = stress_df['total_pnl']
        colors = ['red' if x < 0 else 'green' for x in stress_pnl]
        bars = ax5.bar(range(len(stress_df)), stress_pnl, color=colors, alpha=0.7)
        ax5.set_xticks(range(len(stress_df)))
        ax5.set_xticklabels(stress_df['market_type'], rotation=45, ha='right')
        ax5.set_title('Stress Test Results')
        ax5.set_ylabel('P&L ($)')
        ax5.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # Sensitivity analysis
    if sensitivity_df is not None and not sensitivity_df.empty:
        ax6 = plt.subplot(3, 3, 6)
        ax6.plot(sensitivity_df['confidence_threshold'], sensitivity_df['num_trades'], 
                marker='o', label='Number of Trades', linewidth=2)
        ax6_twin = ax6.twinx()
        ax6_twin.plot(sensitivity_df['confidence_threshold'], sensitivity_df['win_rate'], 
                     marker='s', color='red', label='Win Rate', linewidth=2)
        ax6.set_xlabel('Confidence Threshold')
        ax6.set_ylabel('Number of Trades', color='blue')
        ax6_twin.set_ylabel('Win Rate', color='red')
        ax6.set_title('Parameter Sensitivity Analysis')
        ax6.grid(True, alpha=0.3)
    
    # Performance metrics heatmap
    if not results_df.empty:
        ax7 = plt.subplot(3, 3, 7)
        metrics_for_heatmap = results_df[['win_rate', 'avg_win', 'avg_loss', 'profit_factor']].T
        sns.heatmap(metrics_for_heatmap, annot=True, fmt='.2f', cmap='RdYlGn', 
                   center=0, ax=ax7, cbar_kws={'label': 'Value'})
        ax7.set_title('Performance Metrics Heatmap')
        ax7.set_ylabel('Metrics')
    
    # Risk metrics
    if not results_df.empty:
        ax8 = plt.subplot(3, 3, 8)
        risk_metrics = ['max_drawdown', 'sharpe_ratio', 'profit_factor']
        risk_values = [results_df[metric].mean() for metric in risk_metrics if metric in results_df.columns]
        risk_labels = [metric.replace('_', ' ').title() for metric in risk_metrics if metric in results_df.columns]
        
        if risk_values:
            bars = ax8.bar(risk_labels, risk_values, color=['red', 'blue', 'green'][:len(risk_values)])
            ax8.set_title('Average Risk Metrics')
            ax8.set_ylabel('Value')
            
            # Add value labels
            for bar, value in zip(bars, risk_values):
                ax8.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                        f'{value:.2f}', ha='center', va='bottom')
    
    # Summary statistics table
    if not results_df.empty:
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('off')
        
        summary_stats = [
            ['Total Tests', len(results_df)],
            ['Profitable Days', f"{(results_df['total_pnl'] > 0).sum()}/{len(results_df)}"],
            ['Avg Win Rate', f"{results_df['win_rate'].mean():.1%}"],
            ['Total P&L', f"${results_df['total_pnl'].sum():.2f}"],
            ['Best Day', f"${results_df['total_pnl'].max():.2f}"],
            ['Worst Day', f"${results_df['total_pnl'].min():.2f}"],
            ['Avg Trades/Day', f"{results_df['num_trades'].mean():.1f}"],
            ['Success Rate', f"{(results_df['total_pnl'] > 0).mean():.1%}"]
        ]
        
        table = ax9.table(cellText=summary_stats,
                         colLabels=['Metric', 'Value'],
                         cellLoc='center',
                         loc='center',
                         colWidths=[0.5, 0.5])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        ax9.set_title('Summary Statistics', pad=20)
    
    plt.tight_layout()
    plt.savefig('replay_backtest_comprehensive_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def print_detailed_analysis(results_df, stress_df=None, sensitivity_df=None):
    """Print detailed text analysis of results"""
    
    print("\n" + "="*80)
    print("COMPREHENSIVE REPLAY BACKTEST ANALYSIS")
    print("="*80)
    
    if results_df.empty:
        print("No results to analyze.")
        return
    
    # Overall performance
    total_pnl = results_df['total_pnl'].sum()
    avg_win_rate = results_df['win_rate'].mean()
    profitable_days = (results_df['total_pnl'] > 0).sum()
    total_days = len(results_df)
    
    print(f"\nOVERALL PERFORMANCE:")
    print(f"{'='*30}")
    print(f"Total P&L: ${total_pnl:.2f}")
    print(f"Average Win Rate: {avg_win_rate:.1%}")
    print(f"Profitable Days: {profitable_days}/{total_days} ({profitable_days/total_days:.1%})")
    print(f"Best Day: ${results_df['total_pnl'].max():.2f}")
    print(f"Worst Day: ${results_df['total_pnl'].min():.2f}")
    print(f"Average Trades per Day: {results_df['num_trades'].mean():.1f}")
    
    # Performance by market condition
    print(f"\nPERFORMANCE BY MARKET CONDITION:")
    print(f"{'='*40}")
    market_analysis = results_df.groupby('market_type').agg({
        'win_rate': ['mean', 'std'],
        'total_pnl': ['mean', 'sum'],
        'num_trades': 'mean'
    }).round(3)
    
    for market_type in results_df['market_type'].unique():
        market_data = results_df[results_df['market_type'] == market_type]
        avg_wr = market_data['win_rate'].mean()
        total_pnl = market_data['total_pnl'].sum()
        avg_trades = market_data['num_trades'].mean()
        
        print(f"{market_type:15}: WR={avg_wr:.1%}, P&L=${total_pnl:7.2f}, Trades={avg_trades:.1f}")
    
    # Risk analysis
    if 'max_drawdown' in results_df.columns:
        print(f"\nRISK ANALYSIS:")
        print(f"{'='*20}")
        print(f"Average Max Drawdown: ${results_df['max_drawdown'].mean():.2f}")
        print(f"Worst Drawdown: ${results_df['max_drawdown'].min():.2f}")
        
    if 'sharpe_ratio' in results_df.columns:
        avg_sharpe = results_df['sharpe_ratio'].mean()
        print(f"Average Sharpe Ratio: {avg_sharpe:.2f}")
        
    # Stress test analysis
    if stress_df is not None and not stress_df.empty:
        print(f"\nSTRESS TEST ANALYSIS:")
        print(f"{'='*25}")
        stress_profitable = (stress_df['total_pnl'] > 0).sum()
        stress_total = len(stress_df)
        stress_pnl = stress_df['total_pnl'].sum()
        
        print(f"Stress Test Success Rate: {stress_profitable}/{stress_total} ({stress_profitable/stress_total:.1%})")
        print(f"Stress Test Total P&L: ${stress_pnl:.2f}")
        print(f"Worst Stress Test: ${stress_df['total_pnl'].min():.2f}")
        
    # Sensitivity analysis
    if sensitivity_df is not None and not sensitivity_df.empty:
        print(f"\nSENSITIVITY ANALYSIS:")
        print(f"{'='*22}")
        optimal_conf = sensitivity_df.loc[sensitivity_df['total_pnl'].idxmax(), 'confidence_threshold']
        optimal_pnl = sensitivity_df['total_pnl'].max()
        
        print(f"Optimal Confidence Threshold: {optimal_conf:.2f}")
        print(f"Optimal P&L: ${optimal_pnl:.2f}")
        
        print(f"\nConfidence vs Performance:")
        for _, row in sensitivity_df.iterrows():
            conf = row['confidence_threshold']
            trades = row['num_trades']
            wr = row['win_rate']
            pnl = row['total_pnl']
            print(f"  {conf:.2f}: {trades:2.0f} trades, {wr:.1%} WR, ${pnl:6.2f} P&L")
    
    # Recommendations
    print(f"\nRECOMMENDATIONS:")
    print(f"{'='*20}")
    
    if avg_win_rate < 0.30:
        print("⚠️  Win rate below 30% - consider adjusting model or confidence threshold")
    elif avg_win_rate > 0.40:
        print("✅ Excellent win rate - model performing well")
    else:
        print("✅ Acceptable win rate - within expected range")
    
    if total_pnl > 1000:
        print("✅ Strong profitability - ready for live trading consideration")
    elif total_pnl > 0:
        print("✅ Profitable but modest - consider position sizing optimization")
    else:
        print("⚠️  Unprofitable overall - model needs improvement")
    
    profitable_rate = profitable_days / total_days
    if profitable_rate > 0.60:
        print("✅ High consistency - good daily success rate")
    elif profitable_rate > 0.40:
        print("✅ Reasonable consistency")
    else:
        print("⚠️  Low consistency - high day-to-day variability")
    
    print(f"\nNEXT STEPS:")
    print(f"{'='*15}")
    print("1. Review individual trade logs for pattern analysis")
    print("2. Consider adjusting position sizing based on volatility")
    print("3. Implement additional risk controls if needed")
    print("4. Run forward testing on recent data")
    print("5. Consider paper trading before live implementation")