# udoparser.py

import os
import sys
import logging
from pathlib import Path

# Append the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pdf_parser.semantic_parser import PDFSemanticParser

def setup_logging(logs_dir):
    """
    Configures the logging settings.

    Parameters:
    - logs_dir (str): Directory to store log files.
    """
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    logging.basicConfig(
        filename=os.path.join(logs_dir, 'pdf_parser.log'),
        filemode='a',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )

def main():
    """
    Main function to extract data from PDF and structure it for LLM consumption.
    """
    # Define the path to the PDF file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_filename = 'Charlotte-UDO-as-amended-05.15.2023.pdf'
    pdf_path = os.path.join(current_dir, pdf_filename)

    # Define directories
    output_dir = os.path.join(current_dir, 'output')
    logs_dir = os.path.join(current_dir, 'logs')

    # Setup logging
    setup_logging(logs_dir)
    logging.info("PDF Extraction Started.")

    # Check if the PDF file exists
    if not os.path.exists(pdf_path):
        logging.error(f"PDF file does not exist at path: {pdf_path}")
        print(f"PDF file does not exist at path: {pdf_path}")
        return

    # Initialize parser
    parser = PDFSemanticParser()

    # Parse document
    try:
        parser.parse_document(pdf_path, output_dir)
        logging.info("Extraction completed and data saved to structured_content.json")
        print("Extraction completed and data saved to structured_content.json")
    except Exception as e:
        logging.error(f"Error parsing document: {e}", exc_info=True)
        print(f"Error parsing document: {e}")

if __name__ == "__main__":
    main()
