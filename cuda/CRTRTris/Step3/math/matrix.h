//vector of vectors
//current setup
//contents[row, top to bottom][column]
//      0  1  2  3  4
// 0 [[ x  x  x  x  x ]
// 1  [ x  x  x  x  x ]
// 2  [ x  x  x  x  x ]
// 3  [ x  x  x  x  x ]
// 4  [ x  x  x  x  x ]
// 5  [ x  x  x  x  x ]
// 6  [ x  x  x  x  x ]
// 7  [ x  x  x  x  x ]]


#include <vector>
#include <iostream>
#include <exception>
#include <random>
#include <thread>
#include <algorithm>

using std::vector;
using std::ostream;
using std::endl;
using std::exception;
using std::default_random_engine;
using std::uniform_int_distribution;
using std::initializer_list;
using std::thread;

using std::sort;
using std::reverse;

class sizeMismatchException: public exception
{
	virtual const char* what() const throw()
	{
		return "The matrices' sizes are mismatched";
	}
};

class nonSquareException: public exception
{
	virtual const char* what() const throw()
	{
		return "You can only shear-sort square matrices (at least for now).";
	}
};


template <class T>
class Matrix {

	private:
		int rows;
		int columns;
		vector<vector<T>> *contents;
		default_random_engine *reng;
		enum Orders {Ascending, Descending};
		static bool SortDesc (T i,T j) { return (i>j); }
	
	public:
		//default constructor
		//this looks like a really bad idea
		Matrix() = default;
		
		//rectangular matrix constructor
		//pass num of rows, num of columns, and default value
		Matrix(int inRows, int inCols, T def)
		{
			rows = inRows;
			columns = inCols;
			reng = new default_random_engine();
			contents = new vector<vector<T>>();
			for(int i = 0; i < rows; i++)
			{
				vector<T> row;
				for(int j = 0; j < columns; j++)
				{
					row.push_back(def);
				}
				contents->push_back(row);
			}
		}

		//square matrix constructor
		//pass side length and default value
		Matrix(int size, T def)
		{
			rows = size;
			columns = size;
			reng = new default_random_engine();
			contents = new vector<vector<T>>();
			for(int i = 0; i < rows; i++)
			{
				vector<T> row;
				for(int j = 0; j < columns; j++)
				{
					row.push_back(def);
				}
				contents->push_back(row);
			}
		}

		//square matrix with no specified default value
		//pass side length
		explicit Matrix(int size)
		{
			rows = size;
			columns = size;
			reng = new default_random_engine();
			contents = new vector<vector<T>>();
			for(int i = 0; i < rows; i++)
			{
				vector<T> row(columns);
				contents->push_back(row);
			}
		}
		
		//initlist constructor
		Matrix(int inRows, int inCols, initializer_list<T> args)
		{
			rows = inRows;
			columns = inCols;
			reng = new default_random_engine();
			contents = new vector<vector<T>>();
			
			for(int i = 0; i < rows; i++)
			{
				contents->push_back(vector<T>());
			}
			
			int count = 0;
			for(auto it = begin(args); it != end(args); ++it)
			{
				int index = count/columns;
				((*contents)[index]).push_back(*it);
				count++;
			}
		}
		
		//move constructor
		Matrix(Matrix&& other)
		{
			rows = other.rows;
			columns = other.columns;
			contents = other.contents;
			reng = other.reng;
			
			other.rows = 0;
			other.columns = 0;
			other.contents = NULL;
			other.reng = NULL;
		}
		
		//disable copy constructor
		Matrix(Matrix &c) = delete;
		
		//destructor
		~Matrix()
		{
			delete contents;
			delete reng;
		}
		
		
		//access width
		int GetCols() const { return columns; }
		
		//access height
		int GetRows() const { return rows; }
		
		//set width
		void SetCols(int cols) { columns = cols; }
		
		//set height
		void SetRows(int /*rows*/rowCount) { /*this.*/rows = /*rows*/rowCount; }
		
		//access index in matrix w/ const
		T GetIndex(int row, int column) const { return (*contents)[row][column]; }
		
		//access contents variable
		vector<vector<T>> *Contents() {return contents;}
		
		
		T& operator()(int row, int column) { return (*contents)[row][column]; }
		
		
		//add matrices a + b, place in output
		//check output dim's or set them? prefer latter
//		static void add(const Matrix& a, const Matrix& b, Matrix& output)
//		{
//			if(a.rows != b.rows || a.columns != b.columns/* ||
//				a.rows != output.GetRows() || a.columns != output.GetCols()*/)
//			{
//				sizeMismatchException ex;
//				throw ex;
//			}
//			else
//			{
//				int rows = a.rows;
//				int cols = a.columns;
//				output.SetRows(rows);
//				output.SetCols(cols);
//				for(int row = 0; row < rows; row++)
//				{
//					for(int col = 0; col < cols; col++)
//					{
//						output(row,col) = a.GetIndex(row,col) + b.GetIndex(row,col);
//					}
//				}
//			}
//		}


		

		
		//add matrices a + b, place in output
		//must all be same matrix type, must be called by same matrix type
		static void add(const Matrix& a, const Matrix& b, Matrix& output)
		{
			if(a.GetRows() != b.GetRows() || a.GetCols() != b.GetCols())
			{
				sizeMismatchException ex;
				throw ex;
			}
			else
			{
				int rows = a.GetRows();
				int cols = a.GetCols();
				output.SetRows(rows);
				output.SetCols(cols);
				output.Contents()->clear();
				for(int i = 0; i < rows; i++)
				{
					vector<T> row(cols);
					output.Contents()->push_back(row);
				}
				for(int row = 0; row < rows; row++)
				{
					for(int col = 0; col < cols; col++)
					{
						output(row,col) = a.GetIndex(row,col) + b.GetIndex(row,col);
					}
				}
			}
		}
		
		
		
		void randy(int intput, float f) {std::cout << "randy " << intput << " " << f << std::endl;}
		//add matrices a + b, place in output
		//splitting by row
		static void addThreaded(const Matrix& a, const Matrix& b, Matrix& output, int threadCount)
		{
			if(a.GetRows() != b.GetRows() || a.GetCols() != b.GetCols())
			{
				sizeMismatchException ex;
				throw ex;
			}
			else
			{
				int rows = a.GetRows();
				int cols = a.GetCols();
				output.SetRows(rows);
				output.SetCols(cols);
				output.Contents()->clear();
				for(int i = 0; i < rows; i++)
				{
					vector<T> row(cols);
					output.Contents()->push_back(row);
				}
				
				vector<thread> threads;
				//need object because c++11 threads require it?
				Matrix<T> temp(3);
				
				if(rows < threadCount)
					threadCount = rows;
				
				//all but one thread pull their share
				for(int i = 0; i < threadCount-1; i++)
				{
					// thread name(&Matrix<int>::addAtom, a, b, output, i * (rows / threadCount), rows / threadCount);
					// threads.push_back(thread(&Matrix<T>::randy, &temp, 5, 3.4f));
					threads.push_back(thread(&Matrix<T>::addAtom, &temp, std::ref(a), std::ref(b), std::ref(output), i * (rows / threadCount), rows / threadCount));
					// addAtom(a, b, output, i * (rows / threadCount), rows / threadCount);
				}
				
				//last thread pulls its share and any excess
				// thread name(addAtom, a, b, output, (threadCount-1) * (rows / threadCount), rows / threadCount + rows % threadCount);
				// threads.push_back(thread(&Matrix<T>::randy, &temp, 6, 4.2f));
				threads.push_back(thread(&Matrix<T>::addAtom, &temp, std::ref(a), std::ref(b), std::ref(output), (threadCount-1) * (rows / threadCount), rows / threadCount + rows % threadCount));
				// addAtom(a, b, output, (threadCount-1) * (rows / threadCount), rows / threadCount + rows % threadCount);
				
				for(int i = 0; i < threads.size(); i++)
				{
					threads[i].join();
				}
			}
		}
		
		void addAtom(const Matrix& a, const Matrix& b, Matrix& output, int startRow, int numRows)
		{
			int cols = a.GetCols();
			for(int row = startRow; row < startRow+numRows; row++)
			{
				for(int col = 0; col < cols; col++)
				{
					output(row,col) = a.GetIndex(row,col) + b.GetIndex(row,col);
				}
			}
		}
		
		
		//multiply matrices a * b, places result in output
		//check output dim's or set them? prefer latter
		static void mult(const Matrix& a, const Matrix& b, Matrix& output)
		{
			//a.columns must match b.rows, a.rows x b.columns is the output dimensions
			if(a.GetCols() != b.GetRows())
			{
				sizeMismatchException ex;
				throw ex;
			}
			else
			{
				output.SetRows(a.GetRows());
				output.SetCols(b.GetCols());
				output.Contents()->clear();
				for(int i = 0; i < output.GetRows(); i++)
				{
					vector<T> row(output.GetCols());
					output.Contents()->push_back(row);
				}
				
				for(int i = 0; i < output.GetRows(); i++)
				{
					for(int j = 0; j < output.GetCols(); j++)
					{
						//ith row, jth column is sum of products of a's row i and b's column j
						//start sum off as sum of first row/column, then start at second if exists?
						//concerned about non-declaration
						T sum;
						//this 'must' be true...
						if(a.GetCols() > 0)
							sum = a.GetIndex(i, 0) * b.GetIndex(0, j);
						for(int k = 1; k < a.GetCols(); k++)
						{
							sum = sum + a.GetIndex(i, k) * b.GetIndex(k, j);
						}
						output(i,j) = sum;
					}
				}
				
			}
		}
		
		
		//multiply matrices a * b, places result in output
		//check output dim's or set them? prefer latter
		static void multThreaded(const Matrix& a, const Matrix& b, Matrix& output, int threadCount)
		{
			//a.columns must match b.rows, a.rows x b.columns is the output dimensions
			if(a.GetCols() != b.GetRows())
			{
				sizeMismatchException ex;
				throw ex;
			}
			else
			{
				int rows = a.GetRows();
				int cols = b.GetCols();
				output.SetRows(rows);
				output.SetCols(cols);
				output.Contents()->clear();
				for(int i = 0; i < output.GetRows(); i++)
				{
					vector<T> row(output.GetCols());
					output.Contents()->push_back(row);
				}
				
				vector<thread> threads;
				//need object because c++11 threads require it?
				Matrix<T> temp(3);
				
				if(rows < threadCount)
					threadCount = rows;
				
				//all but one thread pull their share
				for(int i = 0; i < threadCount-1; i++)
				{
					threads.push_back(thread(&Matrix<T>::multAtom, &temp, std::ref(a), std::ref(b), std::ref(output), i * (rows / threadCount), rows / threadCount));
				}
				
				//last thread pulls its share and any excess
				threads.push_back(thread(&Matrix<T>::multAtom, &temp, std::ref(a), std::ref(b), std::ref(output), (threadCount-1) * (rows / threadCount), rows / threadCount + rows % threadCount));
				
				for(int i = 0; i < threads.size(); i++)
				{
					threads[i].join();
				}
				
			}
		}
		
		void multAtom(const Matrix& a, const Matrix& b, Matrix& output, int startRow, int numRows)
		{
			int cols = output.GetCols();
			for(int row = startRow; row < startRow+numRows; row++)
			{
				for(int col = 0; col < cols; col++)
				{
					//rowth row, colth column is sum of products of a's row row and b's column col
					//start sum off as sum of first row/column, then start at second if exists?
					//concerned about non-declaration
					T sum;
					//this 'must' be true...
					if(a.GetCols() > 0)
						sum = a.GetIndex(row, 0) * b.GetIndex(0, col);
					for(int k = 1; k < a.GetCols(); k++)
					{
						sum = sum + a.GetIndex(row, k) * b.GetIndex(k, col);
					}
					output(row,col) = sum;
				}
			}
		}
		
		
		//fill matrix with random numbers
		void rand(unsigned seed)
		{
			reng->seed(seed);
			uniform_int_distribution<long> dist(0, 100);
			for(int i = 0; i < rows; i++)
			{
				for(int j = 0; j < columns; j++)
				{
					(*contents)[i][j] = dist(*reng);
				}
			}
		}
		
		void ShearSort(int numThreads)
		{
			if(rows != columns)
			{
				nonSquareException ex;
				throw ex;
			}
			if(rows == 0 || rows == 1)
				return;
			else
			{
				//this could certainly be better
				if(numThreads > rows)
					numThreads = rows;
				if(numThreads > columns)
					numThreads = columns;
				//sort!
				for(int phase = 0; phase < rows+1; phase++)
				{
					vector<thread> threads;
					
					for(int threadNum = 0; threadNum < (numThreads-1); threadNum++)
					{
						//if even phase, snake, else column
						if(phase % 2 == 0)
						{
							int start = threadNum * (rows / numThreads);
							int end = start + rows / numThreads;
							threads.push_back(thread(&Matrix<T>::ShearRow, this, start, end));
						}
						else
						{
							int start = threadNum * (columns / numThreads);
							int end = start + columns / numThreads;
							threads.push_back(thread(&Matrix<T>::ShearCol, this, start, end));
						}
					}
					
					//last one
					//if even phase, snake, else column
					if(phase % 2 == 0)
					{
						int start = (numThreads-1) * (rows / numThreads);
						int end = start + rows / numThreads + rows % numThreads;
						threads.push_back(thread(&Matrix<T>::ShearRow, this, start, end));
					}
					else
					{
						int start = (numThreads-1) * (columns / numThreads);
						int end = start + columns / numThreads + columns % numThreads;
						threads.push_back(thread(&Matrix<T>::ShearCol, this, start, end));
					}
					
					for(int i = 0; i < threads.size(); i++)
						threads[i].join();
					
				}
			}
		}
		
		void ShearRow(int start, int end)
		{
			for(int row = start; row < end; row++)
			{
				//if even, ascending, else descending
				if(row % 2 == 0)
					// SortRow(row, Orders::Ascending);
					// (*contents)[row] = QuickSort((*contents)[row], Orders::Ascending);
					sort((*contents)[row].begin(), (*contents)[row].end());
				else
				{
					// SortRow(row, Orders::Descending);
					// (*contents)[row] = QuickSort((*contents)[row], Orders::Descending);
					sort((*contents)[row].begin(), (*contents)[row].end(), SortDesc);
					// reverse((*contents)[row].begin(), (*contents)[row].end());
				}
			}
		}
		
		void ShearCol(int start, int end)
		{
			for(int col = start; col < end; col++)
			{
				// SortColumn(col);
				
				vector<T> temp;
				for(int i = 0; i < rows; i++)
					temp.push_back((*contents)[i][col]);
				// temp = QuickSort(temp, Orders::Ascending);
				sort(temp.begin(), temp.end());
				for(int i = 0; i < rows; i++)
					(*contents)[i][col] = temp[i];
			}
		}
		
//		vector<T> QuickSort(vector<T> v, Orders order)
//		{
//			if(v.size() <= 1)
//				return v;
//			T pivot = v[v.size() / 2];
//			v.erase(v.begin() + v.size() / 2);
//			vector<T> less;
//			vector<T> greater;
//			for(int i = 0; i < v.size(); i++)
//			{
//				if(order == Orders::Ascending)
//				{
//					if(v[i] <= pivot)
//						less.push_back(v[i]);
//					else
//						greater.push_back(v[i]);
//				}
//				else //if(order == Orders::Descending)
//				{
//					if(v[i] >= pivot)
//						less.push_back(v[i]);
//					else
//						greater.push_back(v[i]);
//				}
//			}
//			less = QuickSort(less, order);
//			greater = QuickSort(greater, order);
//			less.push_back(pivot);
//			for(int i = 0; i < greater.size(); i++)
//				less.push_back(greater[i]);//could use insert instead to concat
//			return less;
//		}
		
		//slow bubble sort because [reasons]
//		void SortColumn(int col)
//		{
//			//iterate through each item in this column
//			for(int row = 1; row < rows; row++)
//			{
//				//pos is new position for item in row 'row'
//				//if it becomes 0, we know that's where it should be
//				for(int pos = row; pos > 0; pos--)
//				{
//					if((*contents)[pos][col] < (*contents)[pos-1][col])
//					{
//						//swap
//						T temp = (*contents)[pos][col];
//						(*contents)[pos][col] = (*contents)[pos-1][col];
//						(*contents)[pos-1][col] = temp;
//					}
//				}
//			}
//		}
		
		//slow bubble sort because [reasons]
//		void SortRow(int row, Orders order)
//		{
//			//iterate through each item in this row
//			for(int col = 1; col < columns; col++)
//			{
//				//pos is new position for item in column 'col'
//				//if it becomes 0, we know that's where it should be
//				for(int pos = col; pos > 0; pos--)
//				{
//					if(order == Orders::Ascending)
//					{
//						if((*contents)[row][pos] < (*contents)[row][pos-1])
//						{
//							//swap
//							T temp = (*contents)[row][pos];
//							(*contents)[row][pos] = (*contents)[row][pos-1];
//							(*contents)[row][pos-1] = temp;
//						}
//					}
//					else
//					{
//						if((*contents)[row][pos] > (*contents)[row][pos-1])
//						{
//							//swap
//							T temp = (*contents)[row][pos];
//							(*contents)[row][pos] = (*contents)[row][pos-1];
//							(*contents)[row][pos-1] = temp;
//						}
//					}
//				}
//			}
//		}

};

		
//override output operator
template <class T>
ostream& operator<<(ostream& Out, const Matrix<T>& Item)
{
	for(int row = 0; row < Item.GetRows(); row++)
	{
		Out << "[";
		for(int column = 0; column < Item.GetCols() - 1; column++)
		{
			Out << Item.GetIndex(row, column) << " ";
		}
		Out << Item.GetIndex(row, Item.GetCols() - 1);
		Out << "]" << endl;
	}
	return Out;
}