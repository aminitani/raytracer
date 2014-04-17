// ChildView.cpp : implementation of the CChildView class
//

#include "stdafx.h"
#include "lab.h"
#include "ChildView.h"
#include <cmath>
#include <thread>

#include <Windows.h>
#include <iostream>

#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif

using std::thread;

/////////////////////////////////////////////////////////////////////////////
// CChildView

CChildView::CChildView(int width, int height)
{
    SetDoubleBuffer(true);

    m_senorFishyFish.LoadFile(L"models/BLUEGILL.bmp");

	flip = false;
	
	m_fish.SetTexture(&m_senorFishyFish);
	
	m_fish.LoadOBJ("models\\fish4.obj");

	m_width = width;
	m_height = height;
	totThreads = std::thread::hardware_concurrency();
	//totThreads = (std::thread::hardware_concurrency() > 1) ? std::thread::hardware_concurrency()-1 : 1;
	pixels = new float[m_width*m_height*4];
	
	//camera is placed at (0,0,5) and faces in the negative z direction, looking at origin
	float camTrans[16] = {-1.0,0.0,0.0,0.0, 0.0,1.0,0.0,0.0, 0.0,0.0,-1.0,0.0, 0.0,0.0,5.0,1.0};
	camera = new Camera(Transform(camTrans), 3.1415926 * 55.0 / 180.0, (float)m_width/(float)m_height);

	raytracer = new Raytracer(m_width, m_height, pixels, *camera);
	readyToRender = true;
	pendingRending = false;

	SetCursorPos(0, 0);

	mousePos = CPoint(0, 0);

	//m_pDC = NULL;
}

CChildView::~CChildView()
{
	delete camera;
	camera = NULL;
	delete [] pixels;
	pixels = NULL;
	delete raytracer;
	raytracer = NULL;
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

	//SetWindowPos(NULL, (wid > m_width) ? (wid-m_width)/2 : 0, (hit > m_height) ? (hit-m_height)/2 : 0, m_width, m_height, SWP_SHOWWINDOW);
	//SetWindowPos(&wndTop, 550, 218, m_width, m_height, SWP_SHOWWINDOW);
	
	//GLfloat gray = 0.7f;
    //glClearColor(gray, gray, gray, 0.0f);
    //glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glEnable(GL_TEXTURE_2D);

	//glDrawPixels(m_width, m_height, GL_RGBA, GL_FLOAT, pixels);

	glBindTexture(GL_TEXTURE_2D, 1);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, m_width, m_height, 0, GL_RGBA, GL_FLOAT, pixels);

	glClearColor(1,1,1,1);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	//glViewport(0, 0, 

	glBegin(GL_QUADS);
	glTexCoord2f(0.0, 0.0); glVertex3f(-1.0, -1.0, 0.5);
	glTexCoord2f(1.0, 0.0); glVertex3f(1.0, -1.0, 0.5);
	glTexCoord2f(1.0, 1.0); glVertex3f(1.0, 1.0, 0.5);
	glTexCoord2f(0.0, 1.0); glVertex3f(-1.0, 1.0, 0.5);
	glEnd();

	glDisable(GL_TEXTURE_2D);

    //glFlush();
}

void CChildView::Render(int totThreads)
{
	//raytracer->Stop();
	if(readyToRender)
	{
		readyToRender = false;
		raytracer->Render(totThreads, *camera);
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
