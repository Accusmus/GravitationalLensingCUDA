#ifndef ARRAY_H
#define ARRAY_H

// To do:
//
//  - assert number of constructor and operator arguments match
//  - template ndim parameter

// Length array meanings
// <T,1> ncols
// <T,2> nrows, ncols
// <T,3> nslices, nrows, ncols
// <T,4> nblocks, nslices, nrows, ncols

template<typename T, int ndim>
class Array {
public:

  int length[ndim];

  T* buffer;

  int ntotal;

  Array(void) : buffer(NULL) {}

  Array(int ncols) {
    int len[] = {ncols};
    resize(len);
  }
  
  Array(int nrows, int ncols) {
    int len[] = {nrows, ncols};
    resize(len);
  }
  
  Array(int nslices, int nrows, int ncols) {
    int len[] = {nslices, nrows, ncols};
    resize(len);
  }
  
  Array(int nblocks, int nslices, int nrows, int ncols) {
    int len[] = {nblocks, nslices, nrows, ncols};
    resize(len);
  }
  
  Array(int len[]) {
    resize(len);
  }

  void resize(int len[]) {
    ntotal = 1;
    for (int n = 0; n < ndim; ++n) {
      length[n] = len[n];
      ntotal *= length[n];
      //std::cout << length[n] << " " << ntotal << std::endl;
    }

    buffer = new T[ntotal];
    for (int n = 0; n < ntotal; ++n) buffer[n] = 0;
  }

  void reset() {
    for (int n = 0; n < ntotal; ++n) buffer[n] = 0;
  }

  void reset(T value) {
    for (int n = 0; n < ntotal; ++n) buffer[n] = value;
  }

  T& operator [] (int n) {
    return buffer[n];
  }

  T& operator () (int ic) {
    return buffer[ic];
  }

  T& operator () (int ir, int ic) {
    return buffer[ir * length[1] + ic];
  }

  T& operator () (int is, int ir, int ic) {
    return buffer[(is * length[1] + ir) * length[2] + ic];
  }

  T& operator () (int ib, int is, int ir, int ic) {
    return buffer[((ib * length[1] + is) * length[2] + ir) * length[3] + ic];
  }

  ~Array() {
    if (buffer != NULL)
      delete[] buffer;
  }

};

template<typename T>
void resize_A(Array<T, 1>& a, int nc)
{
  int len[] = {nc};
  a.resize(len);
}

template<typename T>
void resize_A(Array<T, 2>& a, int nr, int nc)
{
  int len[] = {nr, nc};
  a.resize(len);
}

template<typename T>
void resize_A(Array<T, 3>& a, int ns, int nr, int nc)
{
  int len[] = {ns, nr, nc};
  a.resize(len);
}

template<typename T>
void resize_A(Array<T, 4>& a, int nb, int ns, int nr, int nc)
{
  int len[] = {nb, ns, nr, nc};
  a.resize(len);
}

#endif
