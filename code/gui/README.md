# GUI Package for Medical Image Analysis

This package provides a comprehensive GUI application for medical image analysis with 3D visualization and model testing capabilities.

## Features

### 1. Data Selection
- Choose between test, validation, and training datasets
- Select specific volumes and slices for analysis
- Real-time data validation and information display

### 2. Model Controls
- Load trained models with configuration verification
- Test model accuracy on selected data
- Process data through loaded models
- Display configuration parameters

### 3. 3D Visualization
- Interactive 3D mesh viewer with VTK
- Multiple view modes: Axial, Sagittal, Coronal, 3D Mesh
- Layer scrolling with slice slider
- Real-time overlay visualization

### 4. Mask Controls
- Toggle visibility for each tumor class (0-3)
- Color-coded masks:
  - Class 0 (Background): Dark Gray #0a0a0a
  - Class 1 (NCR/NET): Red #ff3c3c
  - Class 2 (Edema): Green #3cdc3c
  - Class 3 (ET): Blue #3c78ff
- Semi-transparent overlay support

### 5. Model Processing
- Real-time inference on selected data
- 3D model reconstruction from predictions
- Accuracy metrics calculation
- Per-class performance evaluation

## Installation

1. Install required dependencies:
```bash
pip install -r requirements_gui.txt
```

2. Run the GUI application:
```bash
python run_gui.py
```

## Usage

### Starting the Application
1. Launch the application using `python run_gui.py`
2. The main window will open with control panels on the left and 3D viewer on the right

### Loading a Model
1. Click "Load Model" to load the trained model
2. The configuration will be displayed in the Configuration panel
3. Model checkpoints are automatically loaded from `checkpoints/best_model.pth`

### Selecting Data
1. Choose data split (test/valid/train) from the dropdown
2. Select volume index using the spinbox
3. Choose slice index for 2D viewing
4. Click "Refresh Data" to update available options

### Testing Model Accuracy
1. Load a model first
2. Select desired data split and volume
3. Click "Test Model Accuracy"
4. Results will show overall Dice score and per-class metrics

### Processing Data
1. Load a model and select data
2. Click "Process Selected Data"
3. The 3D viewer will update with predictions
4. Use view mode selector to switch between 2D slices and 3D mesh

### 3D Visualization Controls
- **View Mode**: Choose between Axial, Sagittal, Coronal, or 3D Mesh
- **Display Mode**: Show Image Only, Prediction, Ground Truth, or Overlay
- **Slice Slider**: Navigate through slices in 2D modes
- **Mask Toggles**: Enable/disable specific tumor classes

## File Structure

```
gui/
├── __init__.py           # Package initialization
├── main_window.py        # Main application window
├── data_selector.py      # Data selection component
├── viewer_3d.py          # 3D visualization component
├── model_tester.py       # Model accuracy testing
└── README.md            # This documentation

run_gui.py               # Main application launcher
requirements_gui.txt     # GUI-specific dependencies
```

## Dependencies

- **PyQt5**: GUI framework
- **VTK**: 3D visualization
- **Matplotlib**: Plotting support
- **NumPy**: Array operations
- **PyTorch**: Model inference
- **scikit-learn**: Metrics calculation
- **SimpleITK/nibabel**: Medical image loading

## Troubleshooting

### Common Issues

1. **Import Error**: Make sure all dependencies are installed with `pip install -r requirements_gui.txt`

2. **VTK Rendering Issues**: Update graphics drivers or try software rendering

3. **Model Loading Errors**: Check that `config.json` exists and model checkpoints are in `checkpoints/` directory

4. **Dataset Not Found**: Ensure the dataset path in `config.json` points to the correct location

### Performance Tips

- Use GPU acceleration if available (CUDA)
- Reduce slice range for large volumes
- Disable unused mask overlays for better performance

## API Reference

### MainWindow
Main application window that coordinates all components.

### DataSelector
Component for selecting dataset split, volume, and slice indices.

### Viewer3D
3D visualization component using VTK for rendering.

### ModelTester
Component for evaluating model accuracy and metrics.

## License

This GUI package is part of the Medical Image Analysis project.
