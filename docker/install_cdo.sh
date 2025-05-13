set -e
apt-get install -y libnetcdf-dev libeccodes-dev

cd /tmp

[[ -d cdo-2.4.4 ]] || curl -L https://code.mpimet.mpg.de/attachments/download/29649/cdo-2.4.4.tar.gz | tar xz
cd cdo-2.4.4
./configure --prefix /usr/local --with-netcdf=yes --with-eccodes=yes
make clean
make -j 8 
make install

# clean up
cd /tmp
rm -r cdo-2.4.4
