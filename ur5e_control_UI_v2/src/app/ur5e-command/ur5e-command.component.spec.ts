import { ComponentFixture, TestBed } from '@angular/core/testing';

import { Ur5eCommandComponent } from './ur5e-command.component';

describe('Ur5eCommandComponent', () => {
  let component: Ur5eCommandComponent;
  let fixture: ComponentFixture<Ur5eCommandComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ Ur5eCommandComponent ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(Ur5eCommandComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
