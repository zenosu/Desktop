#!/usr/bin/env python3

import sys
import os
import subprocess
from datetime import datetime

def convert_html_to_pdf_playwright(html_path, pdf_path=None):
    """
    Convert HTML to PDF using Playwright
    """
    try:
        # Try to import playwright
        from playwright.sync_api import sync_playwright
        
        # If no PDF path is provided, create one based on the HTML path
        if not pdf_path:
            pdf_path = os.path.splitext(html_path)[0] + '.pdf'
        
        print(f"Converting {html_path} to PDF using Playwright...")
        
        # Get the absolute path for the HTML file
        abs_html_path = os.path.abspath(html_path)
        file_url = f"file://{abs_html_path}"
        
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.goto(file_url)
            # Wait for any JavaScript to load
            page.wait_for_load_state("networkidle")
            # Create PDF
            page.pdf(path=pdf_path, format="A4", print_background=True)
            browser.close()
        
        print(f"PDF successfully created at: {pdf_path}")
        return True
    except ImportError:
        print("Playwright is not installed. Trying to install it...")
        try:
            # Try to install playwright
            subprocess.run([sys.executable, "-m", "pip", "install", "playwright"], check=True)
            subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"], check=True)
            print("Playwright installed successfully. Please run the script again.")
            return False
        except Exception as e:
            print(f"Error installing Playwright: {e}")
            return False
    except Exception as e:
        print(f"Error converting to PDF with Playwright: {e}")
        return False


def convert_html_to_pdf_simple(html_path, pdf_path=None):
    """
    Use Chrome/Chromium if available to create PDF
    """
    if not pdf_path:
        pdf_path = os.path.splitext(html_path)[0] + '.pdf'
    
    # Check for Chrome or Chromium
    chrome_paths = [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
        "chrome",
        "chromium"
    ]
    
    chrome_path = None
    for path in chrome_paths:
        try:
            if path.startswith("/"):
                # Check if file exists
                if os.path.exists(path):
                    chrome_path = path
                    break
            else:
                # Check if command exists in PATH
                result = subprocess.run(["which", path], capture_output=True, text=True)
                if result.returncode == 0:
                    chrome_path = path
                    break
        except:
            continue
    
    if chrome_path:
        try:
            # Get absolute paths
            abs_html_path = os.path.abspath(html_path)
            abs_pdf_path = os.path.abspath(pdf_path)
            
            print(f"Converting {html_path} to PDF using {chrome_path}...")
            
            # Use Chrome/Chromium to print to PDF
            cmd = [
                chrome_path,
                "--headless",
                "--disable-gpu",
                "--print-to-pdf=" + abs_pdf_path,
                "file://" + abs_html_path
            ]
            result = subprocess.run(cmd, check=True)
            
            if os.path.exists(pdf_path):
                print(f"PDF successfully created at: {pdf_path}")
                return True
            else:
                print("PDF conversion failed.")
                return False
        except Exception as e:
            print(f"Error using Chrome/Chromium to create PDF: {e}")
            return False
    else:
        print("Chrome/Chromium not found.")
        return False


if __name__ == "__main__":
    # If a specific file is provided as argument, use it
    if len(sys.argv) > 1:
        html_path = sys.argv[1]
        pdf_path = sys.argv[2] if len(sys.argv) > 2 else None
    else:
        # Otherwise, try to find the latest report
        reports_dir = "reports"
        html_files = [f for f in os.listdir(reports_dir) if f.endswith('.html')]
        
        if not html_files:
            print(f"No HTML files found in {reports_dir}.")
            sys.exit(1)
        
        # Sort by modification time (newest first)
        latest_html = sorted(html_files, key=lambda x: os.path.getmtime(os.path.join(reports_dir, x)), reverse=True)[0]
        html_path = os.path.join(reports_dir, latest_html)
        pdf_path = os.path.splitext(html_path)[0] + '.pdf'
        
        print(f"Converting latest report: {html_path}")
    
    # Try the conversion methods in order of preference
    if not convert_html_to_pdf_playwright(html_path, pdf_path):
        if not convert_html_to_pdf_simple(html_path, pdf_path):
            print("All PDF conversion methods failed.") 