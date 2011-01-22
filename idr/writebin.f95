! Testing how Fortran handles binary files
!
! Fernando Paolo <fpaolo@ucsd.edu>
! October 27, 2010

program writebin
    implicit none
    integer(4) x
    real(8) y
    logical(1) z
    
    character(100) fname

    integer :: i, n = 10
    
    x = 10
    y = 3.1415 
    z = .true.
    
    call getarg(1, fname) 
    open(10, file=fname, status='unknown', form='unformatted')

    do i = 1, n
        write(10) x, y, z
    enddo

    close(10)
end program writebin
