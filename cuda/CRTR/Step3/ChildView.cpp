// ChildView.cpp : implementation of the CChildView class
//

#include "stdafx.h"
#include "lab.h"
#include "ChildView.h"
#include "testStruct.h"
#include <cmath>
#include <thread>

#include <Windows.h>
#include <iostream>
#include <sstream>
#include <cuda.h>
#include <cuda_runtime_api.h>

//text on screen
#include <stdarg.h>

#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif

using std::thread;

extern "C"
{
void CUDAThrender(float *pixels, TestStruct ts, Camera camera, Scene scene);
void renderTest(float*,int,int);
}

/////////////////////////////////////////////////////////////////////////////
// CChildView

CChildView::CChildView(int width, int height)
{
    SetDoubleBuffer(true);

    //m_senorFishyFish.LoadFile(L"models/BLUEGILL.bmp");
	//m_fish.SetTexture(&m_senorFishyFish);
	//m_fish.LoadOBJ("models\\fish4.obj");

	m_width = width;
	m_height = height;
	totThreads = std::thread::hardware_concurrency();
	//totThreads = (std::thread::hardware_concurrency() > 1) ? std::thread::hardware_concurrency()-1 : 1;
	pixels = new float[m_width*m_height*4];
	for(int i = 0; i < m_width * m_height * 4; i++)
		pixels[i] = 0.0;
	
	//camera is placed at (0,0,5) and faces in the negative z direction, looking at origin
	float camTrans[16] = {-1.0,0.0,0.0,0.0, 0.0,1.0,0.0,0.0, 0.0,0.0,-1.0,0.0, 0.0,0.0,5.0,1.0};
	camera = new Camera(Transform(camTrans), 3.1415926 * 55.0 / 180.0, m_width, m_height);

	scene = new Scene();

	//raytracer = new Raytracer(m_width, m_height, pixels, *camera);
	readyToRender = true;
	pendingRending = false;

	GetCursorPos(&mousePos);

	cudaMalloc((void**)&devPtr, m_width * m_height * sizeof(float) * 4);

	lastFrameTime = std::chrono::high_resolution_clock::now();
	
	BuildFont();

	//m_pDC = NULL;
}

CChildView::~CChildView()
{
	cudaFree((void**)&devPtr);
	delete camera;
	camera = NULL;
	delete [] pixels;
	pixels = NULL;
	//delete raytracer;
	//raytracer = NULL;
	delete scene;
	scene = NULL;

	KillFont();
}


BEGIN_MESSAGE_MAP(CChildView,COpenGLWnd )
    //{{AFX_MSG_MAP(CChildView)
    ON_COMMAND(ID_FILE_SAVEBMPFILE, OnFileSavebmpfile)
    ON_WM_TIMER()
    //}}AFX_MSG_MAP
    ON_WM_LBUTTONDOWN()
    ON_WM_MOUSEMOVE()
    ON_WM_RBUTTONDOWN()
    ON_WM_MOUSEWHEEL()
	ON_COMMAND(ID_RENDER_START, &CChildView::OnRenderStart)
END_MESSAGE_MAP()


/////////////////////////////////////////////////////////////////////////////
// CChildView message handlers

BOOL CChildView::PreCreateWindow(CREATESTRUCT& cs) 
{
    if (!COpenGLWnd::PreCreateWindow(cs))
        return FALSE;

    cs.dwExStyle |= WS_EX_CLIENTEDGE;
    cs.style &= ~WS_BORDER;
    cs.lpszClass = AfxRegisterWndClass(CS_HREDRAW|CS_VREDRAW|CS_DBLCLKS, 
        ::LoadCursor(NULL, IDC_ARROW), HBRUSH(COLOR_WINDOW+1), NULL);

    return TRUE;
}




void CChildView::OnGLDraw(CDC *pDC)
{
    int wid, hit;
    GetSize(wid, hit);
	
	float px = (float)m_width / wid;
	float py = (float)m_height / hit;

	//SetWindowPos(NULL, (wid > m_width) ? (wid-m_width)/2 : 0, (hit > m_height) ? (hit-m_height)/2 : 0, m_width, m_height, SWP_SHOWWINDOW);
	//SetWindowPos(&wndTop, 550, 218, m_width, m_height, SWP_SHOWWINDOW);
	
	//GLfloat gray = 0.7f;
    //glClearColor(gray, gray, gray, 0.0f);
    //glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glEnable(GL_TEXTURE_2D);

	glBindTexture(GL_TEXTURE_2D, 1);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	//rendertest wuz here
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, m_width, m_height, 0, GL_RGBA, GL_FLOAT, pixels);

	glClearColor(0.5, 0.5, 0.5, 1.0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	//glViewport(0, 0, 

	glBegin(GL_QUADS);
	glTexCoord2f(0.0, 0.0); glVertex3f(-px, -py, 0.5);
	glTexCoord2f(1.0, 0.0); glVertex3f(px, -py, 0.5);
	glTexCoord2f(1.0, 1.0); glVertex3f(px, py, 0.5);
	glTexCoord2f(0.0, 1.0); glVertex3f(-px, py, 0.5);
	glEnd();

	glDisable(GL_TEXTURE_2D);

	//high res clock count returns ten millionths of a second
	//auto first = std::chrono::high_resolution_clock::now();
	//Sleep(1000);
	//auto second = std::chrono::high_resolution_clock::now();

	//unsigned long long oneSecond = (second - first).count();

	int fps = 1.0 / ( (std::chrono::high_resolution_clock::now() - lastFrameTime).count() / 10000000.0);
	lastFrameTime = std::chrono::high_resolution_clock::now();
	////DrawText(*pDC, L"sup yo", 10, CRect(50, 50, 100, 50), DT_LEFT);
	//pDC->SetTextAlign(DT_LEFT);
	//pDC->SetTextColor(RGB(255, 255, 255));
	//CRect getRekt;
	//GetClientRect(getRekt);
	//pDC->DrawText("sup yo", getRekt, DT_LEFT);

	std::wstringstream wss;
	wss.fill(' ');
	wss.width(4);
	wss << fps;
	wss << L" FPS";
	GetParentFrame()->SetWindowText(wss.str().c_str());

	//glTranslatef(0.0f,0.0f,-1.0f);
	//glColor3f(1.0, 1.0, 1.0);
	//glRasterPos2f(-.45, 0.0);
	//GLPrint("suuuuuuuuuuuuuuuup");

    //glFlush();
}

//shamelessly ripped out of NEHE gamedev tutorial 13
GLvoid CChildView::BuildFont(GLvoid)								// Build Our Bitmap Font
{
	HFONT	font;										// Windows Font ID
//	HFONT	oldfont;									// Used For Good House Keeping

	base = glGenLists(96);								// Storage For 96 Characters

	font = CreateFont(	-24,							// Height Of Font
						0,								// Width Of Font
						0,								// Angle Of Escapement
						0,								// Orientation Angle
						FW_BOLD,						// Font Weight
						FALSE,							// Italic
						FALSE,							// Underline
						FALSE,							// Strikeout
						ANSI_CHARSET,					// Character Set Identifier
						OUT_TT_PRECIS,					// Output Precision
						CLIP_DEFAULT_PRECIS,			// Clipping Precision
						ANTIALIASED_QUALITY,			// Output Quality
						FF_DONTCARE|DEFAULT_PITCH,		// Family And Pitch
						L"Courier New");					// Font Name

//	oldfont = (HFONT)SelectObject(hDC, font);           // Selects The Font We Want
//	wglUseFontBitmaps(hDC, 32, 96, base);				// Builds 96 Characters Starting At Character 32
//	SelectObject(hDC, oldfont);							// Selects The Font We Want
//	DeleteObject(font);									// Delete The Font
}

GLvoid CChildView::KillFont(GLvoid)
{
	glDeleteLists(base, 96);                // Delete All 96 Characters ( NEW )
}

GLvoid CChildView::GLPrint(const char *fmt, ...)					// Custom GL "Print" Routine
{
	char		text[256];								// Holds Our String
	va_list		ap;										// Pointer To List Of Arguments

	if (fmt == NULL)									// If There's No Text
		return;											// Do Nothing

	va_start(ap, fmt);									// Parses The String For Variables
	    vsprintf(text, fmt, ap);						// And Converts Symbols To Actual Numbers
	va_end(ap);											// Results Are Stored In Text

	glPushAttrib(GL_LIST_BIT);							// Pushes The Display List Bits
	glListBase(base - 32);								// Sets The Base Character to 32
	glCallLists(strlen(text), GL_UNSIGNED_BYTE, text);	// Draws The Display List Text
	glPopAttrib();										// Pops The Display List Bits
}

void CChildView::TurnTable()
{
	//want to rotate camera x degrees around center where x depends on time
	//camera.orientation.forward = (center - eye).normalize();
	//camera.orientation.right = vec3(cos(90deg + angle), 0, -sin(90deg+angle));
	//camera.orientation.up = right cross forward;
}

void CChildView::Render(int totThreads)
{
	if(readyToRender)
	{
		readyToRender = false;
		renderTest(devPtr, m_width, m_height);

		TestStruct ts(1.0, 0.0, 0.0);
		CUDAThrender(devPtr, ts, *camera, *scene);
		cudaDeviceSynchronize();
		cudaMemcpy(pixels, devPtr, m_width * m_height * 4 * sizeof(float), cudaMemcpyDeviceToHost);
		
		Invalidate();
		
		readyToRender = true;

		if(pendingRending)
		{
			thread thrd(&CChildView::Render, this, totThreads);
			thrd.detach();
			pendingRending = false;
		}
	}
	else
		pendingRending = true;
}

double Normal3dv(double *v)
{
    return sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

void CChildView::OnFileSavebmpfile() 
{
    OnSaveImage();
}

void CChildView::OnLButtonDown(UINT nFlags, CPoint point)
{
    //m_camera.MouseDown(point.x, point.y);

    COpenGLWnd ::OnLButtonDown(nFlags, point);
}

void CChildView::OnMouseMove(UINT nFlags, CPoint point)
{
    //if(m_camera.MouseMove(point.x, point.y, nFlags))
	
	if(nFlags & MK_LBUTTON)
	{
		int wid, hit;
		GetSize(wid, hit);

		camera->orientation.Translate( camera->orientation.Left() * ( (point.x - mousePos.x) / (float)wid ) );
		camera->orientation.Translate( camera->orientation.Up() * ( (point.y - mousePos.y) / (float)hit ) );
		thread thrd(&CChildView::Render, this, totThreads);
		thrd.detach();
	}

	mousePos = point;

    COpenGLWnd::OnMouseMove(nFlags, point);
}

void CChildView::OnRButtonDown(UINT nFlags, CPoint point)
{
    //m_camera.MouseDown(point.x, point.y, 2);

    COpenGLWnd::OnRButtonDown(nFlags, point);
}

BOOL CChildView::OnMouseWheel(UINT nFlags, short zDelta, CPoint pt)
{
    //m_camera.MouseWheel(zDelta);

    return COpenGLWnd::OnMouseWheel(nFlags, zDelta, pt);
}

void CChildView::OnKeyDown(UINT nChar, UINT nRepCnt, UINT nFlags)
{
	//switch(nChar)
	//{
	//case 0x44:
	//	break;
	//}

	//COpenGLWnd::OnKeyDown(nChar, nRepCnt, nFlags);
}





void CChildView::OnRenderStart()
{
	//GetSize(m_width, m_height);I break code :)
	//Render((totThreads > 1) ? totThreads-1 : 1);
	thread thrd(&CChildView::Render, this, totThreads);
	thrd.detach();
}

