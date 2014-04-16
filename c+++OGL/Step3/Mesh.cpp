#include "stdafx.h"

#include "Mesh.h"

#include <fstream>      // For input streams from files
#include <string>       // For the string type
#include <sstream>      // For streams from strings

using namespace std;


CMesh::CMesh(void)
{
    m_texture = NULL;
}


CMesh::~CMesh(void)
{
}

void CMesh::Draw()
{
    if(m_texture != NULL)
    {
        glEnable(GL_TEXTURE_2D);
        glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
        glBindTexture(GL_TEXTURE_2D, m_texture->TexName()); 
    }

    glBegin(GL_TRIANGLES);
    for(PTV v=m_triangles.begin();  v!=m_triangles.end();  v++)
    {
		if(m_tvertices.size() > 0)
			glTexCoord2dv(m_tvertices[v->t]);
        glNormal3dv(m_normals[v->n]);
        glVertex3dv(m_vertices[v->v]);
    }
    glEnd();
	
	glDisable(GL_TEXTURE_2D);
}

void CMesh::AddTriangleVertex(int v, int n, int t)
{
	TV tv;
	tv.v = v;
	tv.n = n;
	tv.t = t;
	m_triangles.push_back(tv);
}

// Name :         CMesh::AddFlatQuad()
// Description :  Add a quadrangle to the triangle mesh.
//
void CMesh::AddFlatQuad(int a, int b, int c, int d, int n)
{
    // First triangle
    AddTriangleVertex(a, n, -1);
    AddTriangleVertex(b, n, -1);
    AddTriangleVertex(c, n, -1);

    // Second triangle
    AddTriangleVertex(a, n, -1);
    AddTriangleVertex(c, n, -1);
    AddTriangleVertex(d, n, -1);
}

//
// Name :         CMesh::AddQuad()
// Description :  Add a quadrangle to the triangle mesh with no 
//                assumption of flatness.
//
void CMesh::AddQuad(int a, int b, int c, int d)
{
    // First triangle
    AddTriangleVertex(a, a, -1);
    AddTriangleVertex(b, b, -1);
    AddTriangleVertex(c, c, -1);

    // Second triangle
    AddTriangleVertex(a, a, -1);
    AddTriangleVertex(c, c, -1);
    AddTriangleVertex(d, d, -1);
}


// Name :         CMesh::ComputeSmoothNormals()
// Description :  Compute a normal for each vertex that is the
//                average of the surface normals incident on the 
//                vertex.
//
void CMesh::ComputeSmoothNormals()
{
    m_normals.resize(m_vertices.size());
    for(unsigned int i=0;  i<m_vertices.size();  i++)
        m_normals[i] = CGrVector(0, 0, 0, 0);


    for(PTV v=m_triangles.begin();  v!=m_triangles.end();  )
    {
        // Collect up the vertices of a triangle...
        int a = v->v;
        v++;
        int b = v->v;
        v++;
        int c = v->v;
        v++;

        // Surface normal
        CGrVector normal = Cross(m_vertices[b] - m_vertices[a], 
                                 m_vertices[c] - m_vertices[a]);
        normal.Normalize3();

        // Add to the incident vertices normals
        m_normals[a] += normal;
        m_normals[b] += normal;
        m_normals[c] += normal;
	}

	// Normalize the normals
	for(unsigned int i=0;  i<m_vertices.size();  i++)
		m_normals[i].Normalize3();
}

// Name :         CMesh::LoadOBJ()
// Description :  Load a file in OBJ format.
//
void CMesh::LoadOBJ(const char *filename)
{
    ifstream str(filename);
    if(!str)
    {
        AfxMessageBox(L"File not found");
        return;
    }

    string line;
    while(getline(str, line))
    {
        istringstream lstr(line);

        string code;
        lstr >> code;
        if(code == "v") 
        {
            double x, y, z;
            lstr >> x >> y >> z;
            AddVertex(CGrVector(x, y, z, 1));
        }
        else if(code == "vn")
        {
			double x, y, z;
			lstr >> x >> y >> z;
			AddNormal(CGrVector(x, y, z, 0.0));
        }
        else if(code == "vt")
        {
			double s, t;
			lstr >> s >> t;
			AddTexCoord(CGrVector(s, t, 0, 1));
        }
        else if(code == "f")
        {
            for(int i=0;  i<3;  i++)
            {
                char slash;
                int v, t, n;
                lstr >> v >> slash >> t >> slash >> n;
                AddTriangleVertex(v-1, n-1, t-1);
            }
        }

    }

}