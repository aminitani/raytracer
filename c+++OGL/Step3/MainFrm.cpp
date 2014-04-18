// MainFrm.cpp : implementation of the CMainFrame class
//

#include "stdafx.h"
#include "lab.h"

#include "MainFrm.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif

/////////////////////////////////////////////////////////////////////////////
// CMainFrame

IMPLEMENT_DYNAMIC(CMainFrame, CFrameWnd)

BEGIN_MESSAGE_MAP(CMainFrame, CFrameWnd)
	//{{AFX_MSG_MAP(CMainFrame)
		// NOTE - the ClassWizard will add and remove mapping macros here.
		//    DO NOT EDIT what you see in these blocks of generated code !
	ON_WM_CREATE()
	ON_WM_SETFOCUS()
	//}}AFX_MSG_MAP
END_MESSAGE_MAP()

static UINT indicators[] =
{
	ID_SEPARATOR,           // status line indicator
	ID_INDICATOR_CAPS,
	ID_INDICATOR_NUM,
	ID_INDICATOR_SCRL,
};

/////////////////////////////////////////////////////////////////////////////
// CMainFrame construction/destruction

CMainFrame::CMainFrame()
{
	// TODO: add member initialization code here
	m_width = 640;
	m_height = 480;
	m_wndView = new CChildView(m_width, m_height);
}

CMainFrame::~CMainFrame()
{
}

int CMainFrame::OnCreate(LPCREATESTRUCT lpCreateStruct)
{
	if (CFrameWnd::OnCreate(lpCreateStruct) == -1)
		return -1;
	// create a view to occupy the client area of the frame
	if (!m_wndView->Create(NULL, NULL, AFX_WS_DEFAULT_VIEW,
		CRect(0, 0, 640, 480), this, AFX_IDW_PANE_FIRST, NULL))
	{
		TRACE0("Failed to create view window\n");
		return -1;
	}
	
	//if (!m_wndToolBar.CreateEx(this, TBSTYLE_FLAT, WS_CHILD | WS_VISIBLE | CBRS_TOP
	//	| CBRS_GRIPPER | CBRS_TOOLTIPS | CBRS_FLYBY | CBRS_SIZE_DYNAMIC) ||
	//	!m_wndToolBar.LoadToolBar(IDR_MAINFRAME))
	//{
	//	TRACE0("Failed to create toolbar\n");
	//	return -1;      // fail to create
	//}

	if (!m_wndStatusBar.Create(this) ||
		!m_wndStatusBar.SetIndicators(indicators,
		  sizeof(indicators)/sizeof(UINT)))
	{
		TRACE0("Failed to create status bar\n");
		return -1;      // fail to create
	}

	// TODO: Delete these three lines if you don't want the toolbar to
	//  be dockable
	//m_wndToolBar.EnableDocking(CBRS_ALIGN_ANY);
	EnableDocking(CBRS_ALIGN_ANY);
	//DockControlBar(&m_wndToolBar);

	return 0;
}

BOOL CMainFrame::PreCreateWindow(CREATESTRUCT& cs)
{
	if( !CFrameWnd::PreCreateWindow(cs) )
		return FALSE;
	// TODO: Modify the Window class or styles here by modifying
	//  the CREATESTRUCT cs



	//cs.dwExStyle &= ~WS_EX_CLIENTEDGE;
	//
	//cs.style &= (0xFFFFFFFF ^ WS_SIZEBOX);
	//cs.style |= WS_BORDER;
	//cs.style &= (0xFFFFFFFF ^ WS_MAXIMIZEBOX);
	cs.dwExStyle &= ~WS_EX_CLIENTEDGE;

	//no work
	//AdjustWindowRectEx(CRect(0, 0, m_width, m_height), cs.dwExStyle, false, cs.dwExStyle);

	cs.lpszClass = AfxRegisterWndClass(0);
	return TRUE;
}

/////////////////////////////////////////////////////////////////////////////
// CMainFrame diagnostics

#ifdef _DEBUG
void CMainFrame::AssertValid() const
{
	CFrameWnd::AssertValid();
}

void CMainFrame::Dump(CDumpContext& dc) const
{
	CFrameWnd::Dump(dc);
}

#endif //_DEBUG

/////////////////////////////////////////////////////////////////////////////
// CMainFrame message handlers
void CMainFrame::OnSetFocus(CWnd* pOldWnd)
{
	//RECT rcClient, rcWind;
	//POINT ptDiff;
	//GetClientRect(&rcClient);
	//GetWindowRect(&rcWind);
	//ptDiff.x = (rcWind.right - rcWind.left) - rcClient.right;
	//ptDiff.y = (rcWind.bottom - rcWind.top) - rcClient.bottom;
	//SetWindowPos(&wndTop, 0, 0, m_width + ptDiff.x, m_height + ptDiff.y, SWP_SHOWWINDOW);

	
	//
	//GetClientRect(&rcClient);
	//GetWindowRect(&rcWind);
	//ptDiff.x = (rcWind.right - rcWind.left) - rcClient.right;
	//ptDiff.y = (rcWind.bottom - rcWind.top) - rcClient.bottom;
	//SetWindowPos(&wndTop, 0, 0, m_width + ptDiff.x, m_height + ptDiff.y, SWP_SHOWWINDOW);
	//
	//GetClientRect(&rcClient);
	//GetWindowRect(&rcWind);
	//ptDiff.x = (rcWind.right - rcWind.left) - rcClient.right;
	//ptDiff.y = (rcWind.bottom - rcWind.top) - rcClient.bottom;
	//SetWindowPos(&wndTop, 0, 0, m_width + ptDiff.x, m_height + ptDiff.y, SWP_SHOWWINDOW);
	//
	//GetClientRect(&rcClient);
	//GetWindowRect(&rcWind);
	//ptDiff.x = (rcWind.right - rcWind.left) - rcClient.right;
	//ptDiff.y = (rcWind.bottom - rcWind.top) - rcClient.bottom;
	//SetWindowPos(&wndTop, 0, 0, m_width + ptDiff.x, m_height + ptDiff.y, SWP_SHOWWINDOW);


	
	//CRect temp(100, 100, m_width, m_height);
	//AdjustWindowRectEx(&temp, this->GetStyle(), true, this->GetExStyle());
	//SetWindowPos(&wndTop, temp.left, temp.top, temp.right, temp.bottom, SWP_SHOWWINDOW);


	// forward focus to the view window
	m_wndView->SetFocus();
}

BOOL CMainFrame::OnCmdMsg(UINT nID, int nCode, void* pExtra, AFX_CMDHANDLERINFO* pHandlerInfo)
{
	// let the view have first crack at the command
	if (m_wndView->OnCmdMsg(nID, nCode, pExtra, pHandlerInfo))
		return TRUE;

	// otherwise, do default handling
	return CFrameWnd::OnCmdMsg(nID, nCode, pExtra, pHandlerInfo);
}

