#!/usr/bin/env python3
"""
Parametric Pattern Application
Main entry point for the application.
"""
import sys
import argparse
from pathlib import Path


def convert_patterns():
    """Convert pattern images to curves."""
    from src.image_to_curves import convert_all_patterns

    print("Converting pattern images to curves...")
    print("Looking for:")
    print("  - BasePattern_Back.png")
    print("  - BasePattern_Front.png")
    print("  - BasePattern_Sleeve.png")
    print()

    try:
        results = convert_all_patterns(
            pattern_dir='.',
            output_dir='BasePatternCurves',
            pixels_per_cm=10
        )

        if results:
            print(f"\nSuccessfully converted {len(results)} patterns!")
            print("Curves saved to BasePatternCurves/")
        else:
            print("\nNo patterns found. Please ensure pattern images are in the current directory.")
            return False

        return True

    except Exception as e:
        print(f"\nError during conversion: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_gui():
    """Run the GUI application."""
    from src.gui import run_gui

    # Check if curves exist
    curves_dir = Path('BasePatternCurves')
    if not curves_dir.exists() or not any(curves_dir.glob('*.json')):
        print("Warning: No curve data found in BasePatternCurves/")
        print("Please run: python main.py --convert")
        print()
        response = input("Would you like to convert patterns now? (y/n): ")
        if response.lower() == 'y':
            if not convert_patterns():
                print("Conversion failed. Please check pattern images and try again.")
                return
            print("\nStarting GUI...")
        else:
            print("Exiting. Run with --convert first.")
            return

    run_gui()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Parametric Pattern Application',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert pattern images to curves (one-time setup)
  python main.py --convert

  # Run the GUI application
  python main.py

  # Or simply
  python main.py
        """
    )

    parser.add_argument(
        '--convert',
        action='store_true',
        help='Convert pattern images to curve data (run this first)'
    )

    args = parser.parse_args()

    if args.convert:
        convert_patterns()
    else:
        run_gui()


if __name__ == '__main__':
    main()
