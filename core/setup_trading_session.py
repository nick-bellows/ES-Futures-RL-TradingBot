#!/usr/bin/env python3
"""
Trading Session Setup
Helps establish Tradovate session to bypass CAPTCHA for paper trading
"""

import os
import webbrowser
import time
import requests
import json
from dotenv import load_dotenv

load_dotenv()

def main():
    """Setup trading session"""
    print("TRADOVATE SESSION SETUP")
    print("=" * 30)
    print()
    
    print("CAPTCHA Workaround Instructions:")
    print("-" * 35)
    print()
    print("1. FIRST - Login via browser:")
    print("   Opening Tradovate demo login page...")
    
    # Open browser to login page
    webbrowser.open("https://demo.tradovateapi.com")
    
    print("\n2. Complete these steps in the browser:")
    print("   - Login with your credentials")
    print("   - Complete any CAPTCHA verification")
    print("   - Wait for dashboard to fully load")
    print()
    
    input("Press ENTER after you've successfully logged in via browser...")
    
    print("\n3. Testing API access...")
    
    # Test authentication
    auth_url = "https://demo.tradovateapi.com/v1/auth/accesstokenrequest"
    auth_data = {
        'name': os.getenv('TRADOVATE_USERNAME', 'your_username'),
        'password': os.getenv('TRADOVATE_PASSWORD', 'your_password'),
        'appId': os.getenv('TRADOVATE_APP_ID', 'ES_RL_Trading_Bot'),
        'appVersion': '1.0',
        'cid': int(os.getenv('TRADOVATE_CID', '0'))
    }
    
    try:
        response = requests.post(auth_url, json=auth_data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            
            if 'accessToken' in result:
                print("✓ SUCCESS: API authentication working!")
                print(f"  Access token received (length: {len(result['accessToken'])})")
                print("\n4. READY FOR PAPER TRADING")
                print("   Run: python start_paper_trading.py")
                return True
                
            elif 'p-captcha' in result:
                print("⚠ CAPTCHA still required")
                print("  Try these additional steps:")
                print("  - Wait 2-3 minutes after browser login")
                print("  - Navigate around the demo platform")
                print("  - Try running this script again")
                return False
            else:
                print(f"? Unexpected response: {result}")
                return False
        else:
            print(f"✗ API Error: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"✗ Connection error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n" + "=" * 50)
        print("SESSION SETUP COMPLETE!")
        print()
        print("Your browser session has bypassed the CAPTCHA.")
        print("You can now run paper trading:")
        print()
        print("  python start_paper_trading.py")
        print()
    else:
        print("\n" + "=" * 50) 
        print("SESSION SETUP INCOMPLETE")
        print()
        print("The CAPTCHA is still active. You may need to:")
        print("- Wait longer after browser login")
        print("- Contact Tradovate for API-only access")
        print("- Use a different authentication method")