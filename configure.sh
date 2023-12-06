mkdir debug;
mkdir release;

cd debug;
cmake -DCMAKE_BUILD_TYPE=Debug ../src;

cd ../release;
cmake -DCMAKE_BUILD_TYPE=Release ../src;

cd ..;
