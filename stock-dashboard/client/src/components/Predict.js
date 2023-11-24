import React, { useState, useContext } from "react";
import StockContext from "../context/StockContext";
import ThemeContext from "../context/ThemeContext";

function Predict() {
    const [predictedDailyValue, setPredictedDailyValue] = useState('');
    const [predictedMonthlyValue, setPredictedMonthlyValue] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const { darkMode } = useContext(ThemeContext);
    const { stockSymbol } = useContext(StockContext);
    const [priceComparison1, setPriceComparison1] = useState('');
    const [priceComparison2, setPriceComparison2] = useState('');

    const handlePredictClick = async () => {
        try {
            setIsLoading(true); // Start loading
            setPredictedDailyValue(''); // Clear previous result
            setPredictedMonthlyValue('');

            console.log("Predict button clicked");
            const response = await fetch(`http://127.0.0.1:5000/predict/${stockSymbol}`);
            
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            const data = await response.json();
            console.log("API response:", data);
            setPredictedDailyValue(data.predictedPrice[0]);
            setPredictedMonthlyValue(data.predictedPrice[1]);
            setPriceComparison1(data.predictedPrice[2]);
            setPriceComparison2(data.predictedPrice[3]);
            console.log("Predicted Value state updated to:", data.predictedPrice);
        } catch (error) {
            console.error('Error fetching prediction:', error);
            setPredictedDailyValue('Error fetching prediction');
            setPredictedMonthlyValue('');
        } finally {
            setIsLoading(false); // Stop loading
        }
    };

    return (
        <div className={`flex flex-col items-center my-4 border-2 rounded-md relative z-50 w-96 ${
            darkMode ? "bg-gray-900 border-gray-800" : "bg-white border-neutral-200"
          }`}>
            <button 
                onClick={handlePredictClick} 
                className="text-lg bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline"
                style={{ fontSize: '1.2rem' }}
            >
                Click to Predict Future Stock Prices
            </button>
            {isLoading ? (
                <p>Loading...</p>
            ) : (
                <>
                    <p>Predicted stock price next day: {predictedDailyValue} {priceComparison1}</p>
                    <p>Predicted stock price next month: {predictedMonthlyValue} {priceComparison2}</p>
                </>
            )}
        </div>
    )
}

export default Predict;
