import React from 'react';

function TextInput(props) {
//   const [value, setValue] = useState(''); // useState hook to manage the input value
    
  const fun = props.setValue;

  return (
    <input
      className={props.className}
      value={props.value}
      onChange={e => fun(e.target.value)}
      placeholder={props.placeholder}
    />
  );
}

export default TextInput;
