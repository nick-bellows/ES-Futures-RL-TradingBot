"""
DataBento to QuantConnect Format Converter

Converts DataBento CSV files to QuantConnect-compatible format for ES futures.
Handles 1-minute OHLCV data with proper timezone conversion.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import logging
from datetime import datetime, timezone
import numpy as np

logger = logging.getLogger(__name__)


class DataBentoConverter:
    """Converts DataBento CSV files to QuantConnect format"""
    
    def __init__(self, source_dir: str, output_dir: str):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def convert_file(self, csv_file: Path) -> pd.DataFrame:
        """Convert a single DataBento CSV file to QC format"""
        logger.info(f"Processing {csv_file.name}")
        
        # Read DataBento CSV
        df = pd.read_csv(csv_file)
        
        # Convert timestamp to datetime
        df['ts_event'] = pd.to_datetime(df['ts_event'])
        
        # Create QC format DataFrame
        qc_df = pd.DataFrame({
            'Time': df['ts_event'].dt.strftime('%Y%m%d %H:%M'),
            'Open': df['open'],
            'High': df['high'], 
            'Low': df['low'],
            'Close': df['close'],
            'Volume': df['volume']
        })
        
        # Sort by time
        qc_df = qc_df.sort_values('Time').reset_index(drop=True)
        
        logger.info(f"Converted {len(qc_df)} records from {csv_file.name}")
        return qc_df
    
    def convert_all_files(self) -> Dict[str, pd.DataFrame]:
        """Convert all CSV files in source directory"""
        csv_files = list(self.source_dir.glob("*.csv"))
        logger.info(f"Found {len(csv_files)} CSV files to convert")
        
        converted_data = {}
        
        for csv_file in csv_files:
            try:
                df = self.convert_file(csv_file)
                
                # Create output filename
                output_name = csv_file.stem + "_qc.csv"
                output_path = self.output_dir / output_name
                
                # Save converted data
                df.to_csv(output_path, index=False)
                converted_data[csv_file.name] = df
                
                logger.info(f"Saved converted data to {output_path}")
                
            except Exception as e:
                logger.error(f"Error processing {csv_file.name}: {e}")
                continue
                
        return converted_data
    
    def combine_data(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Combine all converted data into a single continuous dataset"""
        logger.info("Combining all converted data")
        
        combined_dfs = []
        
        # Sort files by date range in filename
        sorted_files = sorted(data_dict.keys())
        
        for filename in sorted_files:
            df = data_dict[filename].copy()
            df['source_file'] = filename
            combined_dfs.append(df)
            
        combined_df = pd.concat(combined_dfs, ignore_index=True)
        combined_df = combined_df.sort_values('Time').reset_index(drop=True)
        
        # Remove duplicates if any
        combined_df = combined_df.drop_duplicates(subset=['Time'], keep='first')
        
        logger.info(f"Combined dataset has {len(combined_df)} records")
        return combined_df
    
    def create_continuous_contract(self) -> pd.DataFrame:
        """Create continuous ES futures contract from all data"""
        converted_data = self.convert_all_files()
        continuous_df = self.combine_data(converted_data)
        
        # Save continuous contract
        continuous_path = self.output_dir.parent / "continuous" / "ES_continuous_1min.csv"
        continuous_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Drop source_file column for final output
        final_df = continuous_df.drop('source_file', axis=1)
        final_df.to_csv(continuous_path, index=False)
        
        logger.info(f"Saved continuous contract to {continuous_path}")
        return final_df


def main():
    """Main conversion script"""
    logging.basicConfig(level=logging.INFO)
    
    # Paths
    source_dir = "data/databento"
    output_dir = "data/quantconnect"
    
    # Convert data
    converter = DataBentoConverter(source_dir, output_dir)
    continuous_df = converter.create_continuous_contract()
    
    print(f"Conversion complete!")
    print(f"Total records: {len(continuous_df)}")
    print(f"Date range: {continuous_df['Time'].min()} to {continuous_df['Time'].max()}")
    
    return continuous_df


if __name__ == "__main__":
    main()