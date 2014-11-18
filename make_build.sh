rm -rf ./builds/"predictor_build_-$(date +"%Y-%m-%d").zip" 
zip ./builds/"predictor_build-$(date +"%Y-%m-%d").zip" ./predictor/*.py ./predictor/*.txt ./predictor/*.csv
./predictor/test_results/

