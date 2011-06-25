!
! Read 100-Byte IDR files: Seasat, Geosat, GFO and ERS-1/-2
!
! * convert MJD to time in seconds since 1-Jan-1985 (aka utc85 or ESA time)
! * unapply (add) tide correction, for later use of better ones
! * apply selected increment for orbit correction
! * check flags for 'retracked' and 'problem retracking'
! * filter out points with undefined elevation values
! * filter out points with invalid lat/lon values
! * filter out points with unavailable geophysical corrections
! * save to ASCII or Binary format (same_original_name.txt or .bin)
! * save output files to specified output directory
! * all arguments are passed trough the command line
! 
! Usage
! -----
! $ gfortran -fconvert=big-endian readidr_ra1.f90 -o readidr_ra1 
! $ ./readidr_ra1 -h
!
! Notes
! -----
! Undefined surf values are set to -9999
! Undefined corrections are set to 32767
! For GFO use inc: 2 (better inc) or 1 (more points)
! For Geosat/GM: use inc 2
! For Seasat, Geosat/ERM, ERS-1/-2: use inc 3
!
! For IDR format see:
! http://icesat4.gsfc.nasa.gov/data_products/level2.php
!
! For examples on how to use the code see:
! http://fspaolo.net/code
!
! Obs
! ---
! This program follows the Fortran 2003 standard.
!
! Fernando Paolo <fpaolo@ucsd.edu>
! December, 2009

! 100-Byte IDR structure for version 4 
module readidr_ra1
   implicit none
   type IDRv4      
      character(2) id
      integer(2) retstat1
      integer(4) time
      integer(4) lat
      integer(4) lon
      integer(4) surf
      integer(4) wdr
      integer(4) altim
      integer(4) altstat
      integer(4) surfstat
      integer(2) iono
      integer(2) wet1
      integer(2) dry
      integer(2) geoid
      integer(2) stide 
      integer(2) otide 
      integer(2) slope 
      integer(2) swh
      integer(2) agc
      integer(2) att
      integer(2) empty1
      integer(2) inc1
      integer(2) empty2
      integer(2) inc2
      integer(2) empty3
      integer(2) inc3
      integer(2) retramp1
      integer(2) retramp2
      integer(2) sigramp1
      integer(2) sigramp2
      integer(2) tanslope
      integer(2) empty4
      integer(2) wet2
      integer(2) modstat
      integer(2) locstat
      integer(2) rshstat
      integer(2) wwlstat
      integer(2) opstat
      integer(2) thres10 
      integer(2) thres20
      integer(2) thres50
      integer(2) retstat2
   endtype   
end module

program main

   use readidr_ra1
   implicit none

   type(IDRv4) :: idr

   character(2) :: id
   real(8) :: secrev, fsecrev, secdat, mjd, fday, utc85
   real(8) :: lat, lon, surf, otide, agc, inc 
   integer(4) :: orbit, surfstat, surfcheck
   integer(2) :: retstat, ionocheck, wetcheck, drycheck
   integer(2) :: stidecheck, otidecheck, slopecheck, inccheck
   integer(2) :: fret, fprob, fmode, fotide

   character(100) :: arg, infile, outfile, outdir, incr
   integer :: nfiles, npts, nvalidpts, nrec, nopt, ios, i
   logical :: filecreated, ascii, verbose

   integer, parameter :: RECLEN = 100      ! record length in bytes
   real(8), parameter :: MJD85 = 46066.    ! 1-Jan-1985 00:00:00h in MJD
   real(8), parameter :: DAYSECS = 86400.  ! 1 day in seconds 

   ascii = .true.; nopt = 0; outdir = 'None'; incr = '3'; verbose = .false.
   call get_arguments()

   print '(a,i9,a)', 'processing files: ', command_argument_count()-nopt, ' ...'
 
   ! iterate over input files
   nfiles = 0; npts = 0; nvalidpts = 0
   do i = nopt+1, command_argument_count() 

      call get_command_argument(i, infile)  ! get arg(i) -> infile
      open(1, file=infile, status='old', access='direct', &
             form='unformatted', recl=RECLEN)

      if (verbose) print '(a,a)', 'file: ', infile
 
      ! iterate over records
      filecreated = .false.; nrec = 1; ios = 0
      do 
         read(1, rec=nrec, iostat=ios) idr
         if (ios /= 0) exit            ! if EOF
         nrec = nrec + 1
         id = idr%id
         
         ! IDR Rev Record
         if (id == 'IR') then   
            !-------------------------------------------------------
            orbit = idr%time            ! orbit number
            mjd = idr%lat               ! days (integer part) 
            secrev = idr%lon            ! secs (integer part)
            fsecrev = idr%surf / 1e6    ! secs (fractional part)
            !-------------------------------------------------------

         ! IDR data Record
         else if (id == 'ID') then
            !-------------------------------------------------------
            secdat = idr%time / 1e6     ! secs (since time in Rev)
            lat = idr%lat / 1e6         ! latitude (deg)
            lon = idr%lon / 1e6         ! longitude (deg)
            surf = idr%surf / 1e2       ! surface elevation (m)
            otide = idr%otide / 1e3     ! ocean tide correction (m)
            agc = idr%agc / 1e2         ! automatic gain control (dB)

            select case (incr)
               case ('1')
                  inc = idr%inc1 / 1e2  ! orbit correction (m)
                  inccheck = idr%inc1   ! check orbit increment
               case ('2')
                  inc = idr%inc2 / 1e2     
                  inccheck = idr%inc2      
               case default
                  inc = idr%inc3 / 1e2     
                  inccheck = idr%inc3      
            endselect

            retstat = idr%retstat1      ! retracking status flags (15-0)
            surfstat = idr%surfstat     ! surface status flags (31-0)
            surfcheck = idr%surf        ! check whether surf is valid 
            ionocheck = idr%iono        ! check ionosphere
            wetcheck = idr%wet1         ! check wet tropo
            drycheck = idr%dry          ! check dry tropo
            stidecheck = idr%stide      ! check solid tide
            otidecheck = idr%otide      ! check ocean tide
            slopecheck = idr%slope      ! check slope corr
            !-------------------------------------------------------

            ! Flags: big-end: 0-15 (lsb) -> little-end: 15-0 (msb)
            !-------------------------------------------------------
            if (btest(retstat, 13) .or. &   ! wvfm spec shaped: 0=no, 1=yes
                btest(retstat, 11) .or. &   ! wvfm spec retracked: 0=no, 1=yes
                btest(retstat, 8)   .or. &  ! problem w/leading edge: 0=no, 1=yes
                btest(retstat, 7)) then 
               fprob = 1                    ! problem retracking: 0=no, 1=yes
            else
               fprob = 0
            endif
            if (btest(retstat, 5)) then     ! wvfm retracked: 0=no, 1=yes
               fret = 1                     
            else
               fret = 0                                          
            endif
            if (btest(retstat, 0)) then     ! mode: 0=ocean, 1=ice
               fmode = 1   
            else
               fmode = 0                           
            endif
            if (btest(surfstat, 7)) then    ! otide cor applied: 0=no, 1=yes
               fotide = 1 
            else
               fotide = 0
            endif
            !-------------------------------------------------------
            
            npts = npts + 1
 
            !!! filter and compute

            ! filter values within correct range
            if (.not. ( -90 <= lat .and. lat <= 90 .and. &
                       -180 <= lon .and. lon <= 360) ) cycle

            if ( (abs(lat) < 1e-6 .and. lat /= 0.) .or.  &
                 (abs(lon) < 1e-6 .and. lon /= 0.) ) cycle

            ! select pts with available geophysical corr
            if (&
                surfcheck /= -9999 .and. &
                ionocheck /= 32767 .and. &
                wetcheck /= 32767 .and. &
                drycheck /= 32767 .and. &
                stidecheck /= 32767 .and. &
                inccheck /= 32767 &
                !slopecheck /= 32767 &
                ) then
                  
               nvalidpts = nvalidpts + 1

               ! fday: is "fraction" of a day
               ! mjd: is modified julian "days"
               ! utc85: is seconds passed since 1-Jan-1985 0:0:0h
               fday = (secrev + fsecrev + secdat) / DAYSECS   ! days
               utc85 = (mjd + fday - MJD85) * DAYSECS         ! secs
 
               ! add increment and detide
               surf = surf + inc 
               if (fotide == 1) surf = surf + otide
 
               !!! output
 
               if (ascii .and. .not. filecreated) then 
                  if (outdir /= 'None') then
                     call split_path()
                     outfile = trim(outdir) // '/' // trim(infile) // '.txt'
                  else
                     outfile = trim(infile) // '.txt'
                  endif
                  open(2, file=outfile, status='unknown', form='formatted')
                  filecreated = .true.
                  nfiles = nfiles + 1
               else if (.not. filecreated) then
                  if (outdir /= 'None') then
                     call split_path()
                     outfile = trim(outdir) // '/' // trim(infile) // '.bin'
                  else
                     outfile = trim(infile) // '.bin'
                  endif
                  open(2, file=outfile, status='unknown', form='unformatted')
                  filecreated = .true.
                  nfiles = nfiles + 1
               endif
 
               if (lon < 0) lon = lon + 360.          ! output lon: 0/360
 
               if (ascii) then
                  write(2, '(i6, f20.6, 2f14.6, 2f14.3, 3i2)') &           ! [edit]
                     orbit, utc85, lat, lon, surf, agc, fmode, fret, fprob
               else
                  ! all to binary double precision
                  write(2) &                          
                     orbit, utc85, lat, lon, surf, agc, fmode, fret, fprob
               endif

            endif
         endif
      enddo
      if (filecreated) close(2)
      close(1)
   enddo

   !print (a,i9), 'number of records:     ', nrec-1
   print '(a,i9)', 'number of points:      ', npts 
   print '(a,i9)', 'number of valid points:', nvalidpts 
   print '(a,i9)', 'files created:         ', nfiles
   if (ascii) then 
      print '(a)', 'output extension: .txt' 
   else
      print '(a)', 'output extension: .bin' 
      print '(a)', 'data in binary format BIG ENDIAN:' 
      print '(a)', 'I*4, F*8, F*8, F*8, F*8, F*8, F*8, I*2, I*2, I*2' 
   endif   

contains

   subroutine get_arguments()
      implicit none
      integer :: k
      logical :: jump = .false.

      if (command_argument_count() < 1) call print_help()

      do k = 1, command_argument_count()  ! iterate over arguments
         if (jump) then                   ! jump one iteration
             jump = .false.
             cycle
         endif
         call get_command_argument(k, arg)

         select case (arg)
            case ('-h', '--help')
               call print_help()
            case ('-v')
               verbose = .true.
               nopt = nopt + 1
            case ('-b')
               ascii = .false.
               print '(a)', 'output format: binary'
               nopt = nopt + 1
            case ('-d')
               call get_command_argument(k+1, outdir)
               if (outdir(:1) == '-' .or. outdir == '') call print_help()
               print '(a,a)', 'output dir: ', trim(outdir)
               nopt = nopt + 2
               jump = .true.
            case ('-i')
               call get_command_argument(k+1, incr)
               if (incr(:1) == '-' .or. incr == '') call print_help()
               print '(a,a)', 'orbit increment: ', trim(incr)
               nopt = nopt + 2
               jump = .true.
            case default
               if (arg(:1) == '-') then 
                  print '(a,a,/)', 'unrecognized command line option: ', trim(arg)
                  call print_help()
               else
                  exit
               endif
         endselect
      enddo
   end subroutine get_arguments

   subroutine split_path()
      implicit none
      integer :: j
      do j = len(infile), 1, -1
         if (infile(j:j) == '/') then 
             infile = trim(infile(j+1:))
             exit
         endif
      enddo
   end subroutine split_path

   subroutine print_help()
      print '(a)', 'usage: ./readidr_ra1 [-h] [-v] [-b] [-i 1|2|3] [-d /output/dir] file1 file2 ...'
      print '(a)', ''
      print '(a)', 'required arguments:'
      print '(a)', '  files       input files to read [ex: /path/to/files/*.ID04]'
      print '(a)', '              note: files always at the end!'
      print '(a)', ''
      print '(a)', 'optional arguments:'
      print '(a)', '  -h, --help  print usage information and exit'
      print '(a)', '  -v          for verbose [default: run silent]'
      print '(a)', '  -b          for binary output files [default: ASCII]'
      print '(a)', '  -i 1|2|3    use orbit increment 1, 2 or 3 [default: 3]'
      print '(a)', '  -d          the output dir [default: same as input file]'
      stop
   end subroutine print_help

end program main
