import React from "react";
import Search from "./Search";
import ThemeIcon from "./ThemeIcon";
import Predict from "./Predict";

const Header = ({ name }) => {
  return (
    <>
      <div className="xl:px-32">
        <h1 className="text-5xl">{name}</h1>
        <div className="flex flex-row justify-between items-center space-x-4">
          <Search />
          <Predict />
        </div>
      </div>
      <ThemeIcon />
    </>
  );
};

export default Header;

// import React, { useState, useContext } from "react";
// import Search from "./Search";
// import ThemeIcon from "./ThemeIcon";
// import StockContext from "../context/StockContext";

// const Header = ({ name }) => {
//   const [predictedValue, setPredictedValue] = useState('');

//   const { stockSymbol } = useContext(StockContext);

//   const handlePredictClick = async () => {
//     try {
//       console.log("Predict button clicked");
//       const response = await fetch(`http://127.0.0.1:5000/predict/${stockSymbol}`);
//       const data = await response.json();
//       console.log("API response:", data);
//       setPredictedValue(data.predictedPrice); // Ensure this matches the API response
//       console.log("Predicted Value state updated to:", data.predictedPrice);
//     } catch (error) {
//       console.error('Error fetching prediction:', error);
//     }
//   };

//   return (
//     <>
//       <div className="xl:px-32 flex justify-between items-center">
//         <div>
//           <h1 className="text-5xl">{name}</h1>
//           <Search />
//         </div>
//         <div>
//           <button onClick={handlePredictClick}>Predict</button>
//           {predictedValue && <p>Predicted Value: {predictedValue}</p>}
//         </div>
//       </div>
//       <ThemeIcon />
//     </>
//   );
// };

// export default Header;
