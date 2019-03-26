const HOST = 'http://127.0.0.1:8080';

const createListItem = function(contents) {
  const list_item = document.createElement('li');
  list_item.appendChild(contents);

  return list_item;
}


const remove_element = function(element) { element.parentElement.removeChild(element); }

const removeVisitor = function(visitor_id) {
  fetch(`${HOST}/api/visitors/del/`, {
    method: 'delete',
    headers: {
      'Accept': 'application/json, text/plain, */*',
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({visitor_id: visitor_id})
  }).then(res => res.json())
    .then(res => {
      console.log(res);
    });
};

const createVisitor = function(door_id, visitor_id, ...details) {
  const visitor = document.createElement('li');
  visitor.classList.add('visitor');

  const visitor_details = document.createElement('ul');

  for (let i = 0; i < details.length; i++)
    visitor_details.appendChild(createListItem(document.createTextNode(details[i])));

  // Add the visitor key last
  const visitor_id_element = createListItem(document.createTextNode(visitor_id));
  visitor_id_element.classList.add('visitor-id');
  visitor_details.appendChild(visitor_id_element);

  visitor.append(visitor_details);

  const button_container = document.createElement('div')
        copy_button = document.createElement('button')
        delete_button = document.createElement('button');

  copy_button.appendChild(document.createTextNode('C'));

  copy_button.addEventListener('click', _ => {
    // Write a string to the clipboard
    navigator.clipboard.writeText(`${HOST}/visitor.html?door=${door_id}&visitor=${visitor_id}`)
      .then(
        _ => { window.alert('Visitor URL copied to clipboard'); },
        _ => { window.alert('Failed copying URL to clipboard!'); });
  });


  delete_button.appendChild(document.createTextNode('X'));

  delete_button.addEventListener('click', _ => {
    if (confirm(`Are you sure you want to delete ${details[0]} (${visitor_id})?`)) {
      removeVisitor(visitor_id);
      remove_element(visitor);
    }
  });

  button_container.appendChild(copy_button);
  button_container.appendChild(delete_button);

  visitor.appendChild(button_container);

  return visitor;
}

const postNewVisitor = function(...detail_obj) {
  fetch(`${HOST}/api/visitors/new/`, {
    method: 'post',
    headers: {
      'Accept': 'application/json, text/plain, */*',
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({name: detail_obj[0]})
  }).then(res => res.json())
    .then(res => {
      const visitor_data = res['data']['visitor'];
      const new_visitor = createVisitor(res['data']['door_id'], visitor_data['key'], visitor_data['name']);
      visitors.appendChild(new_visitor);
    });
}

const visitor_name = document.getElementById('visitor-name')
      visitor_add = document.getElementById('visitor-add')
      visitor_clear = document.getElementById('visitor-clear');

visitor_add.addEventListener('click', () => {
  postNewVisitor(visitor_name.value);
  visitor_name.value = '';
});

visitor_clear.addEventListener('click', () => visitor_name.value = '')

const visitors = document.getElementById('visitor-list');

fetch(`${HOST}/api/visitors/`)
  .then(response => response.json())
  .then(data => {
    const door_id = data['data']['door_id'];
    const visitor_keys = data['data']['visitor_keys'];

    for (let key in visitor_keys) {
      const visitor = createVisitor(door_id, key, visitor_keys[key]);
      visitors.appendChild(visitor);
    };
  });

