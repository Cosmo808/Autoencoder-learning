BUN="UreaPool.\[BUN\]"
creatine="CreatininePool.\[Creatinine(mG/dL)\]"
data_file="../data.txt"

if [[ -e data.txt ]]
then
	rm data.txt
fi
touch data.txt

for category in `ls`
do
	if [[ $category == "BMI"* ]]
	then
		cd $category
		echo -e "#$category\n\n" >> $data_file
		for file in `ls`
		do
			if [[ $file == *".txt"* ]]
			then
				echo -e "#$file：\n\n" >> $data_file
				rows_BUN=$(cat $file | grep $BUN)
				data_BUN=$(echo $rows_BUN | awk '{print $4}')
				echo -e "$BUN: $data_BUN\n" >> $data_file

				rows_crea=$(cat $file | grep $creatine)
				data_crea=$(echo $rows_crea | awk '{print $4}')
				echo -e "$creatine: $data_crea\n\n\n" >> $data_file
			fi
		done
		cd ..
	fi
	echo -e "\n" >> $data_file
done

