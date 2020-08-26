#bash

cd results
for i in result_*
do
    if test -f $i
    then
	rm $i
	echo "Removed $i "
    fi       
done
cd ..

cd variables
for i in variables_*
do
    if test -f $i
    then	
       rm $i
	echo "Removed $i "
    fi       
done
cd ..

cd systems
for i in system_*
do
    if test -f $i
    then
       rm $i
	echo "Removed $i "
    fi       
done
cd ..


cd hsfiles
for i in hsfile_*
do
    if test -f $i
    then
       rm $i
	echo "Removed $i "
    fi       
done
cd ..



