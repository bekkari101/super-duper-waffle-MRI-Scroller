# Medical Image Analysis GUI - Complete Functionality Documentation

## Overview
This GUI application provides a comprehensive interface for medical image analysis, specifically designed for brain tumor segmentation using deep learning models. The application combines data selection, model testing, 3D visualization, and interactive controls into a unified user experience.

## Core Architecture

### Main Window Structure
The application is built around a central main window with two primary panels:
- **Left Panel (Controls)**: Data selection, model controls, mask toggles, and configuration display
- **Right Panel (Visualization)**: 3D viewer with interactive controls

### Component Breakdown

## 1. Data Selection Component (`data_selector.py`)

### Purpose
Handles selection of dataset splits, volumes, and slices for analysis.

### Key Functions
- **Dataset Split Selection**: Choose between 'test', 'valid', and 'train' datasets
- **Volume Selection**: Navigate through available volumes in the selected dataset
- **Slice Selection**: Select specific slices within a volume for 2D viewing
- **Data Validation**: Automatically detects available data and updates UI accordingly
- **Information Display**: Shows dataset statistics and available options

### User Interface Elements
- Dropdown menu for dataset split selection
- Spinbox for volume index selection
- Spinbox for slice index selection  
- Information label showing dataset status
- Refresh button to update data availability

### Data Flow
1. User selects dataset split (test/valid/train)
2. Component scans dataset directory for available volumes
3. Updates volume range based on available data
4. Updates slice range based on selected volume dimensions
5. Provides selected indices to main application

## 2. Model Controls (`main_window.py`)

### Purpose
Manages model loading, configuration verification, and processing operations.

### Key Functions
- **Model Loading**: Loads trained models with checkpoint verification
- **Configuration Display**: Shows current model configuration parameters
- **Accuracy Testing**: Evaluates model performance on selected data
- **Data Processing**: Runs inference on selected data through loaded model
- **Status Management**: Provides real-time feedback on operations

### Model Loading Process
1. User clicks "Load Model" button
2. System validates configuration parameters
3. Creates model instance based on configuration
4. Attempts to load checkpoint from `checkpoints/best_model.pth`
5. Updates UI to reflect model status
6. Enables testing and processing buttons

### Configuration Display
Shows key model parameters including:
- Model type (smp_unet, etc.)
- Number of classes
- Input channels
- Image size
- SMP architecture and encoder

## 3. Model Testing Functionality (`model_tester.py`)

### Purpose
Comprehensive model evaluation with multiple metrics and reporting.

### Testing Capabilities
- **Single Volume Testing**: Evaluate model on specific volume
- **Dataset Testing**: Evaluate model on entire dataset split
- **Per-Class Metrics**: Individual performance for each tumor class
- **Overall Metrics**: Global performance indicators

### Evaluation Metrics
- **Dice Coefficient**: Overlap measure for segmentation accuracy
- **Accuracy**: Overall pixel-wise accuracy
- **Precision**: True positive rate per class
- **Recall**: Sensitivity per class
- **F1 Score**: Harmonic mean of precision and recall

### Testing Process
1. Load selected dataset and volume
2. Run model inference on test data
3. Compare predictions with ground truth
4. Calculate comprehensive metrics
5. Display results in dialog with per-class breakdown

## 4. 3D Visualization Component (`viewer_3d.py`)

### Purpose
Interactive 3D medical image visualization with multiple viewing modes and overlay capabilities.

### Viewing Modes
- **Axial View**: Horizontal slices (top-down view)
- **Sagittal View**: Side-to-side slices
- **Coronal View**: Front-to-back slices
- **3D Mesh Mode**: Full 3D surface reconstruction

### Display Modes
- **Image Only**: Original medical images without overlays
- **Prediction**: Model segmentation predictions only
- **Ground Truth**: Actual segmentation masks only
- **Overlay**: Semi-transparent masks over original images

### Interactive Controls
- **Slice Slider**: Navigate through slices in 2D modes
- **View Mode Selector**: Switch between different anatomical views
- **Display Mode Selector**: Choose visualization type
- **Real-time Updates**: Immediate response to data changes

### 3D Mesh Features
- **Surface Reconstruction**: Creates 3D meshes from segmentation masks
- **Multi-class Rendering**: Different colors for each tumor class
- **Interactive Camera**: Rotate, zoom, pan 3D models
- **Transparency Control**: Adjustable opacity for overlay visualization

## 5. Mask Control System

### Purpose
Interactive control over segmentation mask visibility and appearance.

### Mask Classes
- **Class 0 (Background)**: Non-tumor tissue - Dark Gray (#0a0a0a)
- **Class 1 (NCR/NET)**: Non-enhancing tumor core - Red (#ff3c3c)
- **Class 2 (Edema)**: Swelling around tumor - Green (#3cdc3c)
- **Class 3 (ET)**: Enhancing tumor - Blue (#3c78ff)

### Control Features
- **Toggle Visibility**: Enable/disable individual mask classes
- **Color Coding**: Consistent color scheme for each tumor type
- **Real-time Updates**: Immediate visual feedback when toggling masks
- **Checkbox Interface**: Simple on/off controls for each class

## 6. Data Processing Pipeline

### Purpose
End-to-end processing from data selection to visualization.

### Processing Steps
1. **Data Loading**: Load selected volume and slice data
2. **Preprocessing**: Prepare data for model input (dimension handling, normalization)
3. **Model Inference**: Run data through loaded neural network
4. **Post-processing**: Convert model outputs to segmentation masks
5. **Visualization**: Update 3D viewer with results
6. **Overlay Generation**: Create colored masks for display

### Model Integration
- **Automatic Shape Handling**: Adapts to different input dimensions
- **Batch Processing**: Handles single slices or full volumes
- **Output Conversion**: Transforms model logits to class predictions
- **Memory Management**: Efficient processing of large medical images

## 7. User Workflow

### Typical Usage Pattern
1. **Launch Application**: Start GUI with `python run_gui.py`
2. **Load Model**: Click "Load Model" to initialize neural network
3. **Select Data**: Choose dataset split, volume, and slice indices
4. **Test Model** (Optional): Evaluate accuracy on test data
5. **Process Data**: Run inference to generate predictions
6. **Visualize Results**: Explore predictions in 3D viewer
7. **Adjust Masks**: Toggle mask classes for better visualization
8. **Navigate Views**: Switch between 2D slices and 3D mesh

### Advanced Features
- **Real-time Processing**: Immediate feedback during data selection
- **Configuration Verification**: Model parameters displayed before loading
- **Error Handling**: Comprehensive error messages and recovery
- **Status Updates**: Real-time progress indicators in status bar

## Technical Architecture

### Framework Dependencies
- **PyQt5**: GUI framework for interface components
- **VTK**: 3D visualization and rendering engine
- **PyTorch**: Deep learning model inference
- **NumPy**: Array operations and data handling
- **Matplotlib**: Additional plotting capabilities

### Data Flow Architecture
```
Data Selection → Model Loading → Inference → Visualization
      ↓              ↓            ↓           ↓
Dataset Info → Config Display → Metrics → 3D Rendering
```

### Component Communication
- **Main Window**: Central coordinator for all components
- **Data Selector**: Provides selection indices to main window
- **Model Tester**: Returns evaluation results for display
- **3D Viewer**: Receives processed data for visualization
- **Mask Controls**: Influence visualization rendering in real-time

## File Structure and Organization

### Core GUI Files
- `__init__.py`: Package initialization and version info
- `main_window.py`: Primary application window and coordination
- `data_selector.py`: Dataset navigation and selection interface
- `viewer_3d.py`: 3D visualization and rendering engine
- `model_tester.py`: Model evaluation and metrics calculation

### Supporting Files
- `run_gui.py`: Application launcher and dependency checking
- `requirements_gui.txt`: Complete dependency specifications
- `README.md`: This comprehensive documentation

### Integration Points
- **Config System**: Uses existing `config.py` and `config.json`
- **Model System**: Integrates with `model.py` for model creation
- **Dataset System**: Works with `dataset.py` for data loading
- **Checkpoint System**: Accesses trained models from `checkpoints/`

## Performance Considerations

### Optimization Features
- **Lazy Loading**: Data loaded only when needed
- **Memory Management**: Efficient handling of large 3D volumes
- **GPU Acceleration**: CUDA support for model inference
- **Progressive Rendering**: Step-by-step 3D mesh construction

### Scalability
- **Variable Volume Sizes**: Adapts to different medical image dimensions
- **Flexible Class Numbers**: Configurable for different segmentation tasks
- **Multi-GPU Support**: Potential for distributed processing
- **Cache Management**: Intelligent caching of processed results

## Error Handling and User Feedback

### Robust Error Management
- **Graceful Degradation**: Continues operation despite minor errors
- **Clear Error Messages**: User-friendly error descriptions
- **Recovery Options**: Automatic retry mechanisms where appropriate
- **Status Indicators**: Real-time feedback in status bar

### User Guidance
- **Tool Tips**: Hover information for all controls
- **Status Messages**: Context-aware status updates
- **Dialog Boxes**: Detailed information for complex operations
- **Visual Feedback**: Immediate response to user actions

## Future Enhancement Possibilities

### Potential Extensions
- **Additional View Modes**: MIP (Maximum Intensity Projection) rendering
- **Advanced Metrics**: Hausdorff distance, surface dice calculations
- **Export Capabilities**: Save visualizations as images or videos
- **Batch Processing**: Process multiple volumes automatically
- **Model Comparison**: Side-by-side comparison of different models

### Integration Opportunities
- **DICOM Support**: Direct loading of medical imaging formats
- **Cloud Processing**: Remote model inference capabilities
- **Collaboration Features**: Shared sessions and annotations
- **Advanced Segmentation**: Interactive correction tools

## Summary

This GUI application provides a complete solution for medical image analysis with:
- Intuitive data selection and navigation
- Comprehensive model testing and evaluation
- Advanced 3D visualization with multiple viewing modes
- Interactive mask controls with color-coded tumor classes
- Real-time processing and immediate visual feedback
- Professional medical imaging interface design
- Robust error handling and user guidance
- Extensible architecture for future enhancements

The application successfully bridges the gap between complex deep learning models and practical medical image analysis, providing researchers and clinicians with an accessible yet powerful tool for brain tumor segmentation analysis.
