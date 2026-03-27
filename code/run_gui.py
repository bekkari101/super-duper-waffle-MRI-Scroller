"""
Main GUI launcher for Medical Image Analysis Application - Windows Compatible
"""

import sys
import os
from pathlib import Path

# Set Qt attributes BEFORE creating QApplication
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

# Set high DPI scaling before QApplication creation
QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

# Create QApplication
app = QApplication(sys.argv)
app.setApplicationName("Medical Image Analysis")
app.setApplicationVersion("1.0.0")

def check_dependencies():
    """Check if required dependencies are available"""
    missing_deps = []
    
    # Check PyQt5
    try:
        from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout
        from PyQt5.QtCore import Qt
        print("OK PyQt5 available")
    except ImportError:
        missing_deps.append("PyQt5")
    
    # Check VTK
    try:
        import vtk
        print("OK VTK available")
    except ImportError:
        missing_deps.append("vtk")
    
    # Check NumPy
    try:
        import numpy as np
        print("OK NumPy available")
    except ImportError:
        missing_deps.append("numpy")
    
    # Check PyTorch (with error handling)
    try:
        import torch
        print("OK PyTorch available")
    except ImportError as e:
        print(f"WARNING PyTorch import warning: {e}")
        missing_deps.append("torch")
    except Exception as e:
        print(f"ERROR PyTorch DLL issue: {e}")
        print("Try installing CPU-only PyTorch:")
        print("pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu")
        missing_deps.append("torch")
    
    return missing_deps

def main():
    """Main entry point for the GUI application"""
    print("Starting Medical Image Analysis GUI...")
    
    # Check dependencies
    missing = check_dependencies()
    if missing:
        print(f"\nMissing dependencies: {missing}")
        print("Please install using:")
        print("pip install PyQt5 vtk numpy torch torchvision")
        return
    
    # Add current directory to Python path
    current_dir = Path(__file__).parent
    sys.path.insert(0, str(current_dir))
    
    try:
        # Import GUI components
        print("Loading GUI components...")
        from gui.main_window import MainWindow
        
        # Create and show main window
        window = MainWindow()
        window.show()
        
        print("GUI started successfully!")
        print("\nFeatures available:")
        print("- Data selection (test/valid/train)")
        print("- Volume and slice selection") 
        print("- Model loading and accuracy testing")
        print("- 3D visualization with layer scrolling")
        print("- Mask toggle controls for tumor classes")
        print("- Real-time model processing")
        print("\nUse mouse wheel to scroll through slices!")
        
        # Run the application
        return app.exec_()
        
    except ImportError as e:
        print(f"Error importing GUI modules: {e}")
        print("Make sure all required packages are installed")
        return 1
    except Exception as e:
        print(f"Error starting GUI: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure dataset exists at: DATASET/")
        print("2. Check if config.json exists")
        print("3. Try installing CPU-only PyTorch:")
        print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
