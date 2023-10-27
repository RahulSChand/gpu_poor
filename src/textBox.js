import React from 'react';

function TextInput(props) {
//   const [value, setValue] = useState(''); // useState hook to manage the input value
    
  const fun = props.setValue;
  let disableStatus = false;
  if ('disableStatus' in props){
      disableStatus = props.disableStatus
  }

  return (
    <input
      className={props.className}
      value={props.value}
      onChange={e => { fun(e.target.value); 
        if (typeof props.setChange === 'function') {
          props.setChange(true);
        }
      }}
      onKeyDown={props.handleKeyDown}
      placeholder={props.placeholder}
      disabled={disableStatus}
    />
  );
}

export default TextInput;
