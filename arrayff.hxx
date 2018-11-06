#ifndef ARRAYFF_H
#define ARRAYFF_H

#include "fits.hxx"
#include "array.hxx"

template<typename T, int ndim>
void grab_array(std::string filename, Array<T, ndim>& arr)
{
  FitsFile ff(filename);

  const int ndimff= ff.getAxisDim();

  int len[ndim];
  for (int n = 0; n < ndim; ++n) {
    len[n] = ff.getAxisSize(ndim - n);
  }
  arr.resize(len);
  
  ff.read_data<T>(arr.buffer, arr.ntotal);
  ff.close();
  
}


template<typename T, int ndim>
void dump_array(Array<T, ndim>& arr, std::string filename)
{
  FitsFile fout;
  fout.create_file(filename);

  long fpixel[ndim];
  long naxes[ndim];
  for (int n = 0; n < ndim; ++n) {
    naxes[n] = arr.length[ndim - n - 1];
    fpixel[n] = 1;
  }

  int status = 0;
  int bpix = bitpix<T>();
  if (fits_create_img(fout.fptr, bpix, ndim, naxes, &status))
    FitsFile::handle_error(status);

  long npixels = arr.ntotal;
  int dtype = datatype<T>();
  status = 0;
  if (fits_write_pix(fout.fptr, dtype, fpixel, npixels, arr.buffer, &status))
    FitsFile::handle_error(status);

  fout.close();   
}



#endif
