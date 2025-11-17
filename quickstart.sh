#!/bin/bash

# Quick Start Script for News-Stock Sentiment Algorithm

echo "================================================"
echo "News-Stock Sentiment Analysis - Quick Start"
echo "================================================"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Check for .env file
if [ ! -f ".env" ]; then
    echo ""
    echo "Creating .env file from template..."
    cp .env.example .env
    echo ""
    echo "⚠️  IMPORTANT: Edit .env file and add your API keys!"
    echo "   Get WorldNewsAPI key from: https://worldnewsapi.com/"
    echo ""
    echo "   Then run: python main.py"
    echo ""
else
    echo ""
    echo ".env file already exists"
fi

# Optional: Download spaCy model
echo ""
echo "Do you want to download spaCy model for better entity extraction? (y/n)"
read -r response
if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    echo "Downloading spaCy model..."
    python -m spacy download en_core_web_sm
fi

# Create data directory
mkdir -p data experiments/results reports/charts

echo ""
echo "================================================"
echo "Setup Complete!"
echo "================================================"
echo ""
echo "Next steps:"
echo "1. Edit .env and add your WorldNewsAPI key"
echo "2. Run: python main.py"
echo ""
echo "Or run with sample data (no API key needed):"
echo "   python main.py --symbols AAPL MSFT"
echo ""
echo "To run experiments:"
echo "   python experiment_runner.py --suite weights"
echo ""
echo "For help:"
echo "   python main.py --help"
echo ""
