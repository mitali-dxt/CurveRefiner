# CurveRefiner
CurveRefiner is an innovative project designed to visualize 2D curves represented as polylines. It focuses on enhancing the curves by considering key properties such as regularization, symmetry, and completeness.

# Objective
The project is designed to work with curves represented by cubic Bézier curves and polylines, with outputs visualized in CSV format.


# Features
## Regularize Curves
- Straight Lines: Detect straight lines within the curve data.
- Circles and Ellipses: Identify circular shapes where all points are equidistant from a center or ellipses with two focal points.
- Rectangles and Rounded Rectangles: Distinguish between sharp-edged and rounded rectangles.
- Regular Polygons: Detect polygons with equal sides and angles.
- Star Shapes: Identify star-shaped curves with a central point and radial arms.

## Exploring Symmetry in Curves
-  Detect and analyze the symmetry present in curves, including both reflectional and rotational symmetry.

## Completing Incomplete Curves
- Fully Contained Shapes: One shape completely inside another.
- Partially Contained Shapes: Shapes that are connected but partially occluded.
- Disconnected Shapes: Curves that are fragmented due to occlusion.

# Outcome
## Regularization of the input curve
![Screenshot 2024-08-11 232241](https://github.com/user-attachments/assets/4c0c7cde-1472-42df-8806-b9f271a1aa90)

## Detection of symmetry in the input curve
![Screenshot 2024-08-11 232258](https://github.com/user-attachments/assets/1eefabf4-6c67-48e2-9cc8-81fe0f978a3c)

## Auto Completion of the input curve
![Screenshot 2024-08-11 220328](https://github.com/user-attachments/assets/8daa8232-7672-4b24-9243-581bc5619f2b)


# Tech & Algorithms
- OpenCV
- Mathematical Equations
- Linear Regression
- Python and its libraries





