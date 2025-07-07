import java.io.*;
import java.util.*;
/**
 *
 * class for part of speech HMM algorithm
 * trains on data and tags parts of speech for files or user inputs
 *
 * @author Shad Hassan, Feb 15
 *
 */
public class partOfSpeech {
    public Map<String, Map<String, Double>> transitions;
    public Map<String, Map<String, Double>> observations;

    //constructor instantiates obs and tag maps by taking in file names for training data
    public partOfSpeech(String trainingTags, String trainingObservations) throws Exception{
        transitions = new HashMap<String, Map<String, Double>>();
        observations = new HashMap<String, Map<String, Double>>();
        try{
            //trains maps on inputted training data using training methods
            trainTransition(trainingTags);
            trainObservations(trainingObservations, trainingTags);
        }

        //training method will throw exception if there is an error.
        catch (Exception e) {
            throw new Exception(e);
        }
    }

    //method to train tag transitions, takes in file name for tag training data.
    public void trainTransition(String inputTagFile) throws FileNotFoundException {
        BufferedReader inputTag = new BufferedReader(new FileReader(inputTagFile));
        String tagLine;
        try{
            //read every line on text file
            while((tagLine = inputTag.readLine()) != null){

                //split every line into a list of tags
                String[] tags = tagLine.split(" ");

                //create probabilities by tallying up each transition starting with
                //# as the start for every sentence
                String current = "#";
                for (String tag : tags){

                    //checks if the current tag has already been seen, if not adds it to the transition
                    //map with an empty map for its next tags
                    if(!transitions.containsKey(current)){
                        transitions.put(current, new HashMap<>());
                    }
                    //if its first time this current tag has a transition to the next tag,
                    //create transition with value 0
                    if(!transitions.get(current).containsKey(tag)){
                        transitions.get(current).put(tag, 1.0);
                    }
                    //update tally by one everytime the current tag transitions to a specific next tag.
                    else{
                        transitions.get(current).put(tag, transitions.get(current).get(tag) + 1.0);
                    }
                    //increment along the sentence
                    current = tag;
                }
            }
            //call normalizing helper method and close-buffered reader
            normalizing(transitions);
            inputTag.close();
        }
        //catch any exceptions thrown by buffered reader
        catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    //method to get probably of a specific tag to a specific word in a map of maps
    public void trainObservations(String inputObsFile, String inputTagFile) throws FileNotFoundException {

        //use buffer readers to open the file with senteces and file with tags for those sentences
        BufferedReader inputTag = new BufferedReader(new FileReader(inputTagFile));
        BufferedReader inputObs = new BufferedReader(new FileReader(inputObsFile));
        String obsLine;
        String tagLine;

        try{

            //loop over every line in tag and observation training data
            while((tagLine = inputTag.readLine()) != null
                    && ((obsLine = inputObs.readLine()) != null)){
                //convert every line into a list
                String[] tags = tagLine.split(" ");
                String[] obs = obsLine.split(" ");

                //go through every word and tally which words correspond to which tags
                for (int index = 0; index < obs.length; index++ ){
                    String word = obs[index].toLowerCase();
                    String tag = tags[index];
                    if (!observations.containsKey(tag)){
                        observations.put(tag, new HashMap<>());
                    }
                    if(!observations.get(tag).containsKey(word)){
                        observations.get(tag).put(word, 0.0);
                    }
                    observations.get(tag).put(word, observations.get(tag).get(word) + 1.0);
                }
            }
            //use normalizing helper method to get probabilities and close-buffered readers
            normalizing(observations);
            inputObs.close();
            inputTag.close();
        }
        //catches any exposition thrown by buffered readers
        catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    //helper method iterates over all values of a map of map and normalizes value
    //by tracking total transitions and taking the total occurrences of a specific
    // tag and diving by the total transitions and logging this value
    public void normalizing(Map<String, Map<String, Double>> map){
        for(String curr: map.keySet()){
            Double sum = 0.0;
            for(String next: map.get(curr).keySet()){
                sum += map.get(curr).get(next);
            }
            for(String next: map.get(curr).keySet()){
                map.get(curr).put(next, Math.log(map.get(curr).get(next)/sum));
            }
        }
    }


    //viterbi method to return a list of best tags for a given list of observation
    public String[] viterbi(String observationLine) {
        //create a list of observations from inputted string
        String[] observationList = observationLine.split(" ");

        //set to track current states, map to track current state and scores to next states,
        //backtrack list to track the map of highest scores for each state and transition
        Set<String> currStates = new HashSet<>();
        Map<String, Double> currentScores = new HashMap<>();
        ArrayList<Map<String, String>> backTrackList = new ArrayList<>();

        //add starting state to current states and current scores.
        currStates.add("#");
        currentScores.put("#", 0.0);

        //iterate over all observations
        for(int i = 0; i < observationList.length; i ++){

            //map tracking next states score from the current state and adding
            //a new map for each state to backtrack list
            Map<String, Double> nextScores = new HashMap<>();
            backTrackList.add(new HashMap<>());

            //iterate over all current states
            for(String currentState: currentScores.keySet()){

                //if tag isn't in transitions, ignore
                if(!transitions.containsKey(currentState)){
                    continue;
                }

                //iterate over all next states from the current state
                for(String nextState: transitions.get(currentState).keySet()){
                    //calculate the score for each transition based on its transition score, and
                    //if the tag contains next observations, using -100 as a penalty if not.
                    Double nextScore;
                    if(observations.get(nextState).containsKey(observationList[i])){
                        nextScore = transitions.get(currentState).get(nextState) +
                                currentScores.get(currentState) + observations.get(nextState).get(observationList[i]);
                    }
                    else{
                        nextScore = transitions.get(currentState).get(nextState) +
                                currentScores.get(currentState) + -100.0;
                    }
                    if(!nextScores.containsKey(nextState) || nextScore > nextScores.get(nextState)){
                        nextScores.put(nextState, nextScore);
                        backTrackList.get(i).put(nextState, currentState);
                    }
                }
            }
            //update current scores to next scores
            currentScores = nextScores;
        }

        //find the best tag in the last current scores map, should be the tail
        //of best tag sequences
        Double bestScore = 1.0;
        String bestTag = null;
        for(String tag: currentScores.keySet()){
            if (bestTag == null){
                bestTag = tag;
            }
            if(bestScore > 0 || bestScore < currentScores.get(tag)){
                bestScore = currentScores.get(tag);
                bestTag = tag;
            }
        }

        //trace back through backtrack list of maps logging each state and going to where
        //it came from
        String currTag = bestTag;
        String[] tags = new String[observationList.length];
        //tags[observationList.length - 1] = currTag;
        for (int i = observationList.length - 1; i > -1; i --){
            tags[i] = currTag;
            String nextTag = backTrackList.get(i).get(currTag);
            currTag = nextTag;
        }

        //return in an order list of tags for a given observations
        return tags;
    }

    //method that allows a user to type in a sentence and get its parts of
    //speech tags in the user interface
    public void interactive(){

        //runs Viterbi method on an input sentence from used
        System.out.println("Type in test sentence: ");
        Scanner scanner = new Scanner(System.in);
        String input = scanner.nextLine();
        System.out.println("Original sentence: " + input);
        String[] tagList = viterbi(input);
        String output = " ";

        //turn returned list into a string
        for(String tag: tagList){
            output += " " + tag;
        }
        System.out.println("Translated to tags: " + output);
    }

    //method to test the accuracy of Viterbi method
    public String testMethod(String testSentences, String testTags) throws FileNotFoundException {

        //use buffered readers to open test files
        BufferedReader testSent = new BufferedReader(new FileReader(testSentences));
        BufferedReader testTag = new BufferedReader(new FileReader(testTags));

        //keep track of correct and incorrect tags
        int correct = 0;
        int wrong = 0;
        String testSentLine;
        String testTagLine;
        try {
            //use viterbi on each line of a test and compare output with actual tags,
            //keeping track of correct and incorrect tags.
            while((testSentLine = testSent.readLine()) != null && (testTagLine = testTag.readLine()) != null){
                String[] viterbiOutput = viterbi(testSentLine);
                String[] comparisonTags = testTagLine.split(" ");
                for(int i = 0; i < viterbiOutput.length; i ++){
                    if(viterbiOutput[i].equals(comparisonTags[i])){
                        correct += 1;
                    }
                    else{
                        wrong += 1;
                    }
                }
            }
        }
        //catches any exposition thrown by buffered readers
        catch (IOException e) {
            throw new RuntimeException(e);
        }

        //return message displaying accuracy of Viturbi on a test given file
        double percent = (1 - (double) wrong /correct) * 100;
        return "Correct: " + correct + ", Wrong: " + wrong + ", Percent correct: " + percent ;
    }

    public static void main(String[] args) throws Exception {

        //tests using sample data
        partOfSpeech test1 = new partOfSpeech("PS5/simple-train-tags.txt", "PS5/simple-train-sentences.txt");
        System.out.println(test1.testMethod("PS5/simple-test-sentences.txt", "PS5/simple-test-tags.txt" ));
        System.out.println(Arrays.toString(test1.viterbi("trains are fast .")));
        System.out.println(Arrays.toString(test1.viterbi("my dog bark is beautiful .")));

        //tests using brown data
        partOfSpeech test2 = new partOfSpeech("PS5/brown-train-tags.txt", "PS5/brown-train-sentences.txt");
        System.out.println(Arrays.toString(test1.viterbi("trains are fast .")));
        System.out.println(Arrays.toString(test1.viterbi("my dog bark is beautiful .")));
        System.out.println(test2.testMethod("PS5/brown-test-sentences.txt", "PS5/brown-test-tags.txt" ));
        System.out.println(test2.testMethod("PS5/example-sentences.txt", "PS5/example-tags.txt"));
        test2.interactive();
        test1.interactive();
    }
}