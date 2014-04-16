// ChildView.cpp : implementation of the CChildView class
//

#include "stdafx.h"
#include "lab.h"
#include "ChildView.h"
#include <cmath>
#include <thread>
#include <chrono>

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

CChildView::CChildView()
{
    SetDoubleBuffer(true);

    m_senorFishyFish.LoadFile(L"models/BLUEGILL.bmp");

	flip = false;
	
	m_fish.SetTexture(&m_senorFishyFish);
	
	m_fish.LoadOBJ("models\\fish4.obj");

	m_width = 680;
	m_height = 480;
	totThreads = std::thread::hardware_concurrency();
	//totThreads = (std::thread::hardware_concurrency() > 1) ? std::thread::hardware_concurrency()-1 : 1;
	pixels = new float[m_width*m_height*4];
	
	//camera is placed at (0,0,5) and faces in the negative z direction, looking at origin
	float camTrans[16] = {-1.0,0.0,0.0,0.0, 0.0,1.0,0.0,0.0, 0.0,0.0,-1.0,0.0, 0.0,0.0,5.0,1.0};
	camera = new Camera(Transform(camTrans), 3.1415926 * 55.0 / 180.0, (float)m_width/(float)m_height);

	raytracer = new Raytracer(m_width, m_height, pixels, *camera);
	readyToRender = true;
	pendingRending = false;

	//m_pDC = NULL;
}

CChildView::~CChildView()
{
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
	//GLfloat gray = 0.7f;
    //glClearColor(gray, gray, gray, 0.0f);
    //glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    int wid, hit;
    GetSize(wid, hit);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glDrawPixels(m_width, m_height, GL_RGBA, GL_FLOAT, pixels);

    glFlush();

	//m_pDC = pDC;

	//Invalidate();
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
	camera->orientation.Translate(raytracer->GetCamera()->orientation.Left() * -1);
	thread thrd(&CChildView::Render, this, totThreads);
	thrd.detach();

    COpenGLWnd ::OnLButtonDown(nFlags, point);
}

void CChildView::OnMouseMove(UINT nFlags, CPoint point)
{
    //if(m_camera.MouseMove(point.x, point.y, nFlags))

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
