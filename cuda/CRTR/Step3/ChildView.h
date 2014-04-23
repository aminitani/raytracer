// ChildView.h : interface of the CChildView class
//
/////////////////////////////////////////////////////////////////////////////

#if !defined(AFX_CHILDVIEW_H__50C8628F_112A_40D0_9AB2_53368988C69B__INCLUDED_)
#define AFX_CHILDVIEW_H__50C8628F_112A_40D0_9AB2_53368988C69B__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

#include <chrono>
#include "graphics/OpenGLWnd.h"
#include "graphics/GrTexture.h"	// Added by ClassView
#include "Mesh.h"
#include "raytracer.h"
#include "scene.h"
#include "assets\camera.h"

/////////////////////////////////////////////////////////////////////////////
// CChildView window

class CChildView : public COpenGLWnd
{
// Construction
public:
	//CChildView();

// Attributes
public:

// Operations
public:

// Overrides
	// ClassWizard generated virtual function overrides
	//{{AFX_VIRTUAL(CChildView)
	protected:
	virtual BOOL PreCreateWindow(CREATESTRUCT& cs);
	//}}AFX_VIRTUAL

// Implementation
public:
	void OnGLDraw(CDC *pDC);
	CChildView(int width, int height);
	virtual ~CChildView();

	// Generated message map functions
protected:
	//{{AFX_MSG(CChildView)
	afx_msg void OnFileSavebmpfile();
    afx_msg void OnLButtonDown(UINT nFlags, CPoint point);
    afx_msg void OnMouseMove(UINT nFlags, CPoint point);
    afx_msg void OnRButtonDown(UINT nFlags, CPoint point);
    afx_msg BOOL OnMouseWheel(UINT nFlags, short zDelta, CPoint pt);
	afx_msg void OnKeyDown(UINT nChar, UINT nRepCnt, UINT nFlags);
    //}}AFX_MSG
	DECLARE_MESSAGE_MAP()
private:
	//CGrTexture m_senorFishyFish;

	//int m_mode;
	
	//CMesh m_fish;
	
	float *devPtr;
	float *pixels;
	Raytracer *raytracer;
	Scene *scene;
	Camera *camera;
	
	int m_width;
	int m_height;

	bool readyToRender;
	bool pendingRending;//if you ask to render while it's rendering, make note so we render again after finished (there may be no other stimulus to render since we ignore requests while rendering

	CPoint mousePos;

	std::chrono::high_resolution_clock::time_point lastFrameTime;
	GLuint  base;

	bool useGPU;

	float tTAngle;//angle of the turntable spin
	UINT_PTR tTTimer;//timer
	
	GLvoid BuildFont(GLvoid);
	GLvoid KillFont(GLvoid);
	GLvoid GLPrint(const char *fmt, ...);
	void Render();
	void TurnTable();
public:
	afx_msg void OnRenderStart();
	afx_msg void OnRenderTurntable();
	afx_msg void OnTimer(UINT_PTR nIDEvent);
	afx_msg void OnUpdateRenderTurntable(CCmdUI *pCmdUI);
	afx_msg void OnComputedeviceGpu();
	afx_msg void OnUpdateComputedeviceGpu(CCmdUI *pCmdUI);
};

/////////////////////////////////////////////////////////////////////////////

//{{AFX_INSERT_LOCATION}}
// Microsoft Visual C++ will insert additional declarations immediately before the previous line.

#endif // !defined(AFX_CHILDVIEW_H__50C8628F_112A_40D0_9AB2_53368988C69B__INCLUDED_)
