# Camera Calibration

From OpenCV's Github repo: https://github.com/opencv/opencv/tree/master/samples/python

## Output

After running the sample python program _calibrate.py_ this is the following output:

```bash
Run with 4 threads...
processing ./chessboard/00.jpg...
processing ./chessboard/01.jpg...
processing ./chessboard/03.jpg...
processing ./chessboard/02.jpg...
           ./chessboard/00.jpg... OK
processing ./chessboard/06.jpg...
           ./chessboard/03.jpg... OK
processing ./chessboard/07.jpg...
           ./chessboard/02.jpg... OK
processing ./chessboard/05.jpg...
           ./chessboard/01.jpg... OK
processing ./chessboard/10.jpg...
           ./chessboard/07.jpg... OK
processing ./chessboard/04.jpg...
           ./chessboard/06.jpg... OK
processing ./chessboard/09.jpg...
           ./chessboard/05.jpg... OK
processing ./chessboard/08.jpg...
           ./chessboard/10.jpg... OK
           ./chessboard/04.jpg... OK
           ./chessboard/08.jpg... OK
           ./chessboard/09.jpg... OK

RMS: 1.5482273037472216
camera matrix:
 [[3.65699459e+03 0.00000000e+00 2.29223950e+03]
 [0.00000000e+00 3.64813416e+03 1.12564561e+03]
 [0.00000000e+00 0.00000000e+00 1.00000000e+00]]
distortion coefficients:  [ 0.19278277 -0.74393125  0.00137376 -0.0008844   0.26093983]

Undistorted image written to: ./output/00_undistorted.png
Undistorted image written to: ./output/01_undistorted.png
Undistorted image written to: ./output/03_undistorted.png
Undistorted image written to: ./output/02_undistorted.png
Undistorted image written to: ./output/06_undistorted.png
Undistorted image written to: ./output/07_undistorted.png
Undistorted image written to: ./output/05_undistorted.png
Undistorted image written to: ./output/10_undistorted.png
Undistorted image written to: ./output/04_undistorted.png
Undistorted image written to: ./output/09_undistorted.png
Undistorted image written to: ./output/08_undistorted.png
```