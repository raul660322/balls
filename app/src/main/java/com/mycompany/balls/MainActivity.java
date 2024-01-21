package com.mycompany.balls;

import android.app.*;
import android.os.*;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.objdetect.Objdetect;


import android.app.Activity;
import android.content.Context;
import android.graphics.Color;
import android.media.AudioManager;
import android.media.MediaPlayer;
import android.os.Bundle;
import android.renderscript.Long2;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.TextureView;
import android.view.View;
import android.view.WindowManager;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Toast;
import android.widget.ToggleButton;
import org.opencv.imgcodecs.*;

public class MainActivity extends Activity  implements CvCameraViewListener2{

	private static final String TAG = "OCVSample::Activity";
	private static final Scalar FACE_RECT_COLOR = new Scalar(0, 255, 0, 255);
	public static final int JAVA_DETECTOR = 0;

	Core.MinMaxLocResult mmG;
	Rect eye_only_rectangle;
	Point iris;
	Rect eye_template;

	private int cameraid = 0;
	private Mat templateR;
	private Mat templateL;
	private Mat templateR_open;
	private Mat templateL_open;

	private boolean HaarLE = false;
	private boolean HaarRE = false;
	private boolean HaarEyeOpen_R = false;
	private boolean HaarEyeOpen_L = false;

	private MenuItem mItemFace50;
	private MenuItem mItemFace40;
	private MenuItem mItemFace30;
	private MenuItem mItemFace20;
	private MenuItem mItemType;

	private Mat mRgba;
	private Mat mGray;
	private Rect vRect;
	
	private File mCascadeFile;
	private File cascadeFileER;
	private File cascadeFileEL;
	private File cascadeFileEyeOpen;

	private CascadeClassifier mJavaDetector;
	private CascadeClassifier mJavaDetectorEyeRight;
	private CascadeClassifier mJavaDetectorEyeLeft;
	private CascadeClassifier mJavaDetectorEyeOpen;

	private int mDetectorType = JAVA_DETECTOR;
	private String[] mDetectorName;

	private float mRelativeFaceSize = 0.2f;
	private int mAbsoluteFaceSize = 0;

	private CameraBridgeViewBase mOpenCvCameraView;

	int AllTime = 30;
	int drowsyTime = 1;
	double frequency;
	long timer;
	int TotalFrames = 0;
	int FrameFace = 0;
	int FrameEyesOpen = 0;
	int FrameEyesClosed = 0;
	public int FrameClosedDrowsy = 0;
	boolean flag = false;
	boolean flag_drowsy = false;
	boolean drowsy = true;
	long timer_drowsy;
    int count_drowsy = 0;
	MediaPlayer beep;


//
	private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
		@Override
		public void onManagerConnected(int status) {
			switch (status) {
				case LoaderCallbackInterface.SUCCESS: {
						Log.i(TAG, "OpenCV loaded successfully");
						try {
							// load cascade file from application resources
							//Face detection classifier
							InputStream is = getResources().openRawResource(R.raw.lbpcascade_frontalface);
							File cascadeDir = getDir("cascade", Context.MODE_WORLD_READABLE);
							mCascadeFile = new File(cascadeDir, "lbpcascade_frontalface.xml");
							FileOutputStream os = new FileOutputStream(mCascadeFile);
  							String md=mCascadeFile.getAbsolutePath();
							byte[] buffer = new byte[4096];
							int bytesRead;
							while ((bytesRead = is.read(buffer)) != -1) {
								os.write(buffer, 0, bytesRead);
							}
							is.close();
							os.close();

							// ------------------ load right eye classificator -----------------------
							InputStream iser = getResources().openRawResource(R.raw.haarcascade_righteye_2splits);
							File cascadeDirER = getDir("cascadeER",Context.MODE_PRIVATE);
							cascadeFileER = new File(cascadeDirER,"haarcascade_eye_right.xml");
							FileOutputStream oser = new FileOutputStream(cascadeFileER);

							byte[] bufferER = new byte[4096];
							int bytesReadER;
							while ((bytesReadER = iser.read(bufferER)) != -1) {
								oser.write(bufferER, 0, bytesReadER);
							}
							iser.close();
							oser.close();

							// ------------------ load left eye classificator -----------------------
							InputStream isel = getResources().openRawResource(R.raw.haarcascade_lefteye_2splits);
							File cascadeDirEL = getDir("cascadeEL",Context.MODE_PRIVATE);
							cascadeFileEL = new File(cascadeDirEL,"haarcascade_eye_left.xml");
							FileOutputStream osel = new FileOutputStream(cascadeFileEL);

							byte[] bufferEL = new byte[4096];
							int bytesReadEL;
							while ((bytesReadEL = isel.read(bufferEL)) != -1) {
								osel.write(bufferEL, 0, bytesReadEL);
							}
							isel.close();
							osel.close();

							// ------------------ load open eye classificator -----------------------
							InputStream opisel = getResources().openRawResource(R.raw.haarcascade_eye_tree_eyeglasses);
							File cascadeDirEyeOpen = getDir("cascadeEyeOpen",Context.MODE_PRIVATE);
							cascadeFileEyeOpen = new File(cascadeDirEyeOpen,"haarcascade_eye_tree_eyeglasses.xml");


							FileOutputStream oposel = new FileOutputStream(cascadeFileEyeOpen);

							byte[] bufferEyeOpen = new byte[4096];
							int bytesReadEyeOpen;
							while ((bytesReadEyeOpen = opisel.read(bufferEyeOpen)) != -1) {
								oposel.write(bufferEyeOpen, 0, bytesReadEyeOpen);
							}
							opisel.close();
							oposel.close();

							//Face Classifier 
							Boolean existe=mCascadeFile.exists();
							//File mf=new File("/storage/emulated/0/Cascada/lbpcascade_frontalface.xml");
							//int d= Integer.parseInt(String.valueOf(mf.length()/1024));
							mJavaDetector = new CascadeClassifier(mCascadeFile.getAbsolutePath());
							//mJavaDetector = new CascadeClassifier("/storage/emulated/0/Cascada/lbpcascade_frontalface.xml");
							mJavaDetector.load(mCascadeFile.getAbsolutePath());
							if (mJavaDetector.empty()) {
								Log.e(TAG, "Failed to load cascade classifier of face");
								mJavaDetector = null;
							} else
								Log.i(TAG, "Loaded cascade classifier from "+ mCascadeFile.getAbsolutePath());
							//cascadeDir.delete();

							//EyeRightClassifier
							mJavaDetectorEyeRight = new CascadeClassifier(cascadeFileER.getAbsolutePath());
							mJavaDetectorEyeRight.load(cascadeFileER.getAbsolutePath() );
							if (mJavaDetectorEyeRight.empty()) {
								Log.e(TAG, "Failed to load cascade classifier of eye right");
								mJavaDetectorEyeRight = null;
							} else
								Log.i(TAG, "Loaded cascade classifier from "+ cascadeFileER.getAbsolutePath());
							//cascadeDirER.delete();

							//EyeLeftClassifier
							mJavaDetectorEyeLeft = new CascadeClassifier(cascadeFileEL.getAbsolutePath());
							mJavaDetectorEyeLeft.load( cascadeFileEL.getAbsolutePath() );
							if (mJavaDetectorEyeLeft.empty()) {
								Log.e(TAG, "Failed to load cascade classifier of eye left");
								mJavaDetectorEyeLeft = null;
							} else
								Log.i(TAG, "Loaded cascade classifier from "+ cascadeFileEL.getAbsolutePath());
							//cascadeDirEL.delete();

							//EyeOpenClassifier
							mJavaDetectorEyeOpen = new CascadeClassifier(cascadeFileEyeOpen.getAbsolutePath());
							mJavaDetectorEyeOpen.load(cascadeFileEyeOpen.getAbsolutePath());
							if (mJavaDetectorEyeOpen.empty()) {
								Log.e(TAG, "Failed to load cascade classifier of eye open");
								mJavaDetectorEyeOpen = null;
							} else
								Log.i(TAG, "Loaded cascade classifier from "+ cascadeFileEyeOpen.getAbsolutePath());
							//cascadeDirEyeOpen.delete();

						} catch (IOException e) {
							e.printStackTrace();
							Log.e(TAG, "Failed to load cascade. Exception thrown: " + e);
						}
						mOpenCvCameraView.setCameraIndex(cameraid);
						mOpenCvCameraView.enableFpsMeter();
						mOpenCvCameraView.enableView();
					}
					break;
				default: {
						super.onManagerConnected(status);
					}
					break;
			}
		}
	};

	public MainActivity() {
		mDetectorName = new String[2];
		mDetectorName[JAVA_DETECTOR] = "Java";
		Log.i(TAG, "Instantiated new " + this.getClass());
	}

	/** Called when the activity is first created. */
	@Override
	public void onCreate(Bundle savedInstanceState) {
		Log.i(TAG, "called onCreate");
		super.onCreate(savedInstanceState);
		getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

		setContentView(R.layout.main);

		mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.fd_activity_surface_view);
		mOpenCvCameraView.setCvCameraViewListener(this);

		beep = MediaPlayer.create(this, R.raw.button1);
	}

	@Override
	public void onPause() {
		super.onPause();
		if (mOpenCvCameraView != null)
			mOpenCvCameraView.disableView();
		System.exit(0);
	}

	@Override
	public void onResume() {
		super.onResume();
		//OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_3, this,mLoaderCallback);
		OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_1_0, this,mLoaderCallback);	
	}

	public void onDestroy() {
		super.onDestroy();
		mOpenCvCameraView.disableView();
	}

	public void onCameraViewStarted(int width, int height) {
		mGray = new Mat();
		mRgba = new Mat();
		Point p1=new Point(500,500);
		Point p2=new Point(300,300);
		vRect=new Rect(p1,p2);
	}

	public void onCameraViewStopped() {
		mGray.release();
		mRgba.release();
		//System.exit(0);
	}

	public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
		/*if (drowsy){
			timer_drowsy = Core.getTickCount();
			drowsy = false;
		}*/

		SetTimer();
		mRgba = inputFrame.rgba();
		mGray = inputFrame.gray();
		TotalFrames++;
		
		/*boolean showing_drowsy = SetDrowsy();
		if (showing_drowsy || count_drowsy != 0){
			count_drowsy++;

			Imgproc.putText(mRgba, "ALERT!", new Point(mRgba.size().width/2, mRgba.size().height/2), Core.FONT_HERSHEY_SCRIPT_COMPLEX, 4, new Scalar(255,255,0),5);
			if (count_drowsy>2){count_drowsy=0;}
		}


		if (mAbsoluteFaceSize == 0) {
			int height = mGray.rows();
			if (Math.round(height * mRelativeFaceSize) > 0) {
				mAbsoluteFaceSize = Math.round(height * mRelativeFaceSize);
			}
		}
		MatOfRect faces = new MatOfRect();*/

		mRgba=inputFrame.rgba();
		//Imgproc.rectangle(mRgba, vRect.tl(), vRect.br(),new Scalar(255, 0, 0), 2);
		Mat input= inputFrame.gray();
		//Mat input=i1.submat(vRect);
        Mat circles = new Mat();
        Imgproc.blur(input, input, new Size(10, 10), new Point(2, 2));
        Imgproc.HoughCircles(input, circles, Imgproc.CV_HOUGH_GRADIENT, 2, 2000, 70, 70, 0, 120);

        Log.i(TAG, String.valueOf("size: " + circles.cols()) + ", " + String.valueOf(circles.rows()));

        if (circles.cols() > 0) {
            for (int x=0; x < Math.min(circles.cols(), 5); x++ ) {
                double circleVec[] = circles.get(0, x);

                if (circleVec == null) {
                    break;
                }

                Point center = new Point((int) circleVec[0], (int) circleVec[1]);
                int radius = (int) circleVec[2];

                //Imgproc.circle(mRgba, center, 3, new Scalar(255, 255, 255), 5);
                //Imgproc.circle(input, center, radius, new Scalar(255, 255, 255), 2);
				
				//Point supI=new Point(vRect.tl().x+circleVec[0]-radius,vRect.tl().y+circleVec[1]-radius);
				//Point infD=new Point(vRect.tl().x+circleVec[0]+radius,vRect.tl().y+circleVec[1]+radius);
				//Point sup2I=new Point(vRect.tl().x+circleVec[0]-2*radius,vRect.tl().y+circleVec[1]-2*radius);
				//Point inf2D=new Point(circleVec[0]+2*radius,circleVec[1]+2*radius);
				//Imgproc.rectangle(mRgba, supI, infD,new Scalar(255, 0, 0), 2);
				
				Point supI=new Point(circleVec[0]-radius,circleVec[1]-radius);
				Point infD=new Point(circleVec[0]+radius,circleVec[1]+radius);
				Point sup2I=new Point(circleVec[0]-2*radius,circleVec[1]-2*radius);
				//double a=vRect.x;
				//double b=vRect.y;
				
				//Point sup2I=new Point(a,b);
				//Point inf2D=new Point(300,300);
				Point inf2D=new Point(circleVec[0]+2*radius,circleVec[1]+2*radius);
				
				Imgproc.rectangle(mRgba, supI, infD,new Scalar(255, 0, 0), 2);
				
				vRect=new Rect(sup2I,inf2D);
				Imgproc.rectangle(mRgba, vRect.tl(), vRect.br(),new Scalar(255, 0, 0), 2);
            }
        }
		
		//Trabajar la lineas-------------------
		Mat edges=new Mat();
		Mat linesP=new Mat();
		Mat ii = input.submat(vRect);
		Imgproc.adaptiveThreshold(input, input, 255, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY, 11, 2); 
		//Imgproc.Canny(input, input, 60, 60*3,3,false);
        //Imgproc.HoughLinesP(ii, linesP, 1, Math.PI/180, 60, 60, 10);
		Imgproc.HoughLines(input, linesP, 1, Math.PI/180.0, 100);
		if (linesP.rows() > 0) {
            for (int x=0; x < Math.min(linesP.rows(), 1); x++ ) {
				//double[] data = linesP.get(x, 0);
				//Point p1=new Point(data[0],data[1]);
				//Point p2=new Point(data[2],data[3]);
				double[] data = linesP.get(x, 0);
				double rho = data[0];
				double theta = data[1];
				double a = Math.cos(theta);
				double b = Math.sin(theta);
				double x0 = a*rho;
				double y0 = b*rho;
				//Drawing lines on the image
				Point pt1 = new Point();
				Point pt2 = new Point();
				pt1.x = Math.round(x0 + 100.0*(-b))+vRect.tl().x;
				pt1.y = Math.round(y0 + 100.0*(a))+vRect.tl().y;
				pt2.x = Math.round(x0 - 100.0*(-b))+vRect.tl().x;
				pt2.y = Math.round(y0 - 100.0 *(a))+vRect.tl().y;
				Imgproc.line(mRgba,pt1,pt2,new Scalar(255, 0, 0),3);
			}
		}	
		//-------------------------------------
		//Mat pp=mRgba.submat(vRect);
		//ii.copyTo(pp);
		linesP.release();
		edges.release();
		circles.release();
        //input.release();
        //return inputFrame.rgba();
		return mRgba;
    }
	

	

	@Override
	public boolean onCreateOptionsMenu(Menu menu) {
		Log.i(TAG, "called onCreateOptionsMenu");
		mItemFace50 = menu.add("Face size 50%");
		mItemFace40 = menu.add("Face size 40%");
		mItemFace30 = menu.add("Face size 30%");
		mItemFace20 = menu.add("Face size 20%");
		mItemType = menu.add(mDetectorName[mDetectorType]);
		return true;
	}

	@Override
	public boolean onOptionsItemSelected(MenuItem item) {
		Log.i(TAG, "called onOptionsItemSelected; selected item: " + item);
		if (item == mItemFace50)
			setMinFaceSize(0.5f);
		else if (item == mItemFace40)
			setMinFaceSize(0.4f);
		else if (item == mItemFace30)
			setMinFaceSize(0.3f);
		else if (item == mItemFace20)
			setMinFaceSize(0.2f);
		else if (item == mItemType) {
			int tmpDetectorType = (mDetectorType + 1) % mDetectorName.length;
			item.setTitle(mDetectorName[tmpDetectorType]);
		}
		return true;
	}

	private void setMinFaceSize(float faceSize) {
		mRelativeFaceSize = faceSize;
		mAbsoluteFaceSize = 0;
	}

	public void onToggleClick(View v) {
		cameraid = cameraid^1;
		mOpenCvCameraView.disableView();
	    mOpenCvCameraView.setCameraIndex(cameraid);
	    mOpenCvCameraView.enableView();
    }

	public void InitTimer(View v){
		Toast.makeText(getApplicationContext(), "Timer enabled for "+AllTime+" seconds", Toast.LENGTH_SHORT).show();
		frequency = Core.getTickFrequency(); //freency of the clock. How many clocks cycles per second,
		timer = Core.getTickCount();			//start timer for 1 minute. It gives number of clock cycles.
		TotalFrames = 0;
		FrameFace = 0;
		FrameEyesOpen = 0;
		FrameEyesClosed = 0;
		flag = true;
	}

	public void SetTimer(){
		long newtimer = Core.getTickCount()-timer;
		if(newtimer/frequency>AllTime && flag){
			if(FrameEyesClosed>FrameFace){FrameEyesClosed=FrameFace;}
			if(FrameEyesOpen>FrameFace){FrameEyesOpen=FrameFace;}
			AudioManager audioManager = (AudioManager) getSystemService(Context.AUDIO_SERVICE);
			audioManager.setStreamVolume(AudioManager.STREAM_MUSIC, 10, 0);
			beep.start();
			String msg = "Timer: "+newtimer+" Frecuency: "+(long)frequency;
			final String Result = "Total Frames: "+TotalFrames+"\nFrames face: "+FrameFace+
				"\nFrames EyesOpen: "+FrameEyesOpen+"\nFrames EyesClosed: "+FrameEyesClosed;
			Log.i(TAG, msg);
			Log.i(TAG, Result);

			runOnUiThread(new Runnable() { //Toast crashes when is used gettickcount. So that it is needed
		            public void run(){
		            	Toast.makeText(getApplicationContext(), Result, Toast.LENGTH_LONG).show();
		            }
		        });
			flag = false;
		}
	}

	public boolean SetDrowsy(){
		long newtimer = Core.getTickCount()-timer_drowsy;
		frequency = Core.getTickFrequency();
		flag_drowsy = false;
		if(newtimer/frequency>drowsyTime){
			timer_drowsy = Core.getTickCount();
			if (FrameClosedDrowsy>2){
				flag_drowsy = true;
			}
			FrameClosedDrowsy = 0;
		}
		return flag_drowsy;
	}
}
