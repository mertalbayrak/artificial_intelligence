/****************************************************************************************************************/
/*                                                                                                              */
/*   OpenNN: Open Neural Networks Library                                                                       */
/*   www.artelnics.com/opennn                                                                                   */
/*                                                                                                              */
/*   B L A N K   A P P L I C A T I O N                                                                          */
/*                                                                                                              */
/*   Roberto Lopez                                                                                              */
/*   Artelnics - Making intelligent use of data                                                                 */
/*   robertolopez@artelnics.com                                                                                 */
/*                                                                                                              */
/****************************************************************************************************************/

// System includes

#include <iostream>
#include <math.h>
#include <time.h>

// OpenNN includes

#include "../opennn/opennn.h"

using namespace OpenNN;

int main(void)
{
    try
    {
        int rank =0;
        if(rank == 0)
        {
            std::cout << "OpenNN. Blank Test Application." << std::endl;
        }

        srand((unsigned)time(NULL));

        DataSet dataSet;
        NeuralNetwork neuralNetwork;
        LossIndex lossIndex;
        TrainingStrategy trainingStrategy;

        if(rank == 0)
        {
            dataSet.set_data_file_name("../data/blank.dat");
            dataSet.set_separator("Space");
            dataSet.load_data();

            Variables *variables = dataSet.get_variables_pointer();
            variables->set(8,1);
            variables->set_name(0,"Başörtü takar mı?"); // evet = 1 , hayır = 0
            variables->set_name(1,"Etek giyer mi?");
            variables->set_name(2,"Sakalı var mı, yok mu?");
            variables->set_name(3,"Saçı kısa mı, uzun mu?");
            variables->set_name(4,"Sesi pes mi, tiz mi?");
            variables->set_name(5,"Kıl oranı fazla mı, az mı?");
            variables->set_name(6,"Kas kütlesi fazla mı, az mı?");
            variables->set_name(7,"Ayak ve elleri büyük mü, küçük mü?");
            variables->set_name(8,"erkek veya kadın");

            // Instances

            Instances *instances = dataSet.get_instances_pointer();
            instances->split_random_indices();

            const Matrix<std::string> inputsInformation = variables->arrange_inputs_information();
            const Matrix<std::string> targetsInformation = variables->arrange_targets_information();

            // Neural network
            /*
            const size_t inputsNumber = variables->count_inputs_number();
            const size_t hiddenNeuronsNumber = 15;
            const size_t outputsNumber = variables->count_targets_number();
            */

            //const Vector< Statistics<double> > inputs_statistics = dataSet.scale_inputs_minimum_maximum();
            //const Vector< Statistics<double> > targets_statistics = dataSet.scale_targets_minimum_maximum();

            neuralNetwork.load("../data/neural_network.xml");
            //neuralNetwork.set(inputsNumber,hiddenNeuronsNumber,outputsNumber);
            Inputs* inputs = neuralNetwork.get_inputs_pointer();
            inputs->set_information(inputsInformation);
            Outputs* outputs = neuralNetwork.get_outputs_pointer();
            outputs->set_information(targetsInformation);

            // Loss index
            lossIndex.load("../data/loss_index.xml");
            lossIndex.set_data_set_pointer(&dataSet);
            lossIndex.set_neural_network_pointer(&neuralNetwork);

            // Training strategy
            trainingStrategy.load("../data/training_strategy.xml");
            trainingStrategy.set(&lossIndex);
            trainingStrategy.set_main_type(TrainingStrategy::CONJUGATE_GRADIENT);
            ConjugateGradient* cg = trainingStrategy.get_conjugate_gradient_pointer();
            cg->set_maximum_iterations_number(1000);
            cg->set_reserve_loss_history(true);
            //gdp->set_display_period(100);
            trainingStrategy.perform_training();
        }

        if(rank == 0)
        {
            // Print results to screen
            int j = 0,k = 0,trueCount=0;
            double error=0;
            Vector<double> inputs("../data/input_data.dat");
            Vector<double> temp(8, 0.0);
            Vector<double> output(1, 0.0);
            Vector<double> target("../data/target_data.dat");
            for(int i=0;i<1280;i++)
            {
                if(i%8 != 0)
                {
                    temp[j] = inputs[i];
                    j++;
                }
                else
                {
                    temp[j] = inputs[i];
                    output = neuralNetwork.calculate_outputs(temp);
                    if(output.calculate_binary() == target[k])
                    {
                        std::cout << "Answer: "<< output.calculate_binary() << " True" << std::endl;
                        trueCount++;
                    }
                    else
                        std::cout << "Answer: "<< output.calculate_binary() << " False" << std::endl;
                    k++;
                    j=0;
                }
            }
                error = (160.0-trueCount)*100.0/160.0;
                std::cout << "160 sorudan " << trueCount << " tanesi dogru cevaplanmistir." << std::endl;
                std::cout << "Hata orani: " << error << std::endl;

            // Test Analysis
            TestingAnalysis testing_analysis(&neuralNetwork, &dataSet);

            Matrix<size_t> confusion = testing_analysis.calculate_confusion();

            // Save results
            dataSet.save("../data/data_set.xml");
            neuralNetwork.save("../data/new_neural_network.xml");
            neuralNetwork.save_expression("../data/expression.txt");
            neuralNetwork.save_expression_python("../data/expression.py");
            lossIndex.save("../data/new_loss_index.xml");
            confusion.save("../data/confusion.dat");
            trainingStrategy.save("../data/new_training_strategy.xml");
        }
        return(0);
    }
    catch(std::exception& e)
    {
        std::cout << e.what() << std::endl;

        return(1);
    }
}
