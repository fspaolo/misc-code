! Testing how Fortran handles binary files
!
! Fernando Paolo <fpaolo@ucsd.edu>
! October 27, 2010

program readbin
    implicit none
    type struct
        integer(4) x
        real(8) y
        logical(1) z
    endtype
    type(struct) dat

    character(100) fname

    integer :: ios = 0
    
    call getarg(1, fname) 
    open(10, file=fname, status='old', form='unformatted')

    do 
        read(10, iostat=ios) dat
        if (ios /= 0) exit 
        print *, dat
    enddo

    close(10)
end program readbin
